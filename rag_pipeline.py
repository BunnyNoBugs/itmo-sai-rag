from langchain.llms import HuggingFacePipeline
from langchain_core.runnables.base import RunnableParallel, RunnableLambda, Runnable
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.base import BaseOutputParser
from langchain.output_parsers.retry import RetryOutputParser
from langchain_core.runnables import chain, RunnableParallel, RunnableLambda
from langchain_core.exceptions import OutputParserException
from langchain.docstore.document import Document
from operator import itemgetter
from langchain_core.vectorstores import VectorStoreRetriever
from typing import List
import ast

answer_prompt_template = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Ответь на вопрос по нескольким параграфам. Для ответа используй только информацию в представленных параграфах.
Если на вопрос нельзя ответить исходя из параграфов, напиши \"недостаточно информации для ответа\".
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{context}
Вопрос: {question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Ответ:
"""

hyde_prompt_template = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Напиши абзац на несколько предложений из документа про ГОСТ, отвечающий на вопрос.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Вопрос: {question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Абзац:
"""

planner_prompt_template = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Ты система осуществляющая планирование запросов нужной для пользователя информации в векторной базе с документами разделенными на небольшие сегменты. \
Для каждого поступившего запроса от пользователя \
твоя задача, при необходимости, разделить его на несколько запросов к векторной базе данных для получения нужной для ответа информации. \
Ты должен написать итоговые запросы "Запросы:", а потом написать объяснение "Объяснение:". И запросы и объяснение должны быть на русском языке. \
Например, вопрос "назови столицу России" не нуждается в разделении, необходимая информация о столице России будет извлечена. \
В свою очередь, запрос "Сравни автомобили ауди q5 и audi q7" уже должен быть разделен на два: "Какие преимущества и недостатки и audi q5" и "Какие преимущества и недостатки у audi q7", т.к. нам \
нужна информация по обоим объектам для проведения сравнения. 
Пользуйся следующим набором правил:
1. Если встречается указание на год или день, то учти, что сейчас 14 июня 2024 года и заменяй в итоговых запросах на конкретную информацию для улучшения поиска (например, "в этом году" должно быть изменено на "2024 году", "сегодня" нужно заменить на "14 июня 2024 года"). 
2. Компания для которой ты работаешь и в которой работают пользователи называется Татнефть, где уместно заменяй на конкретное название.
3. Каждый следующий вопрос не должен опираться на результат ответа предыдущего (нельзя: "Какие из этих технологий..."). Агрегация информации произойдет позднее.
4. Компании конкуренты Татнефти в нефтяной области: Газпромнефть, Башнефть, Роснефть, Лукоил.
5. Если будешь отвечать на английском получишь штраф.
6. Учитывай что у пользовательского запроса есть контекст, который нельзя терять.
7. Если пользовательский запрос можно разделить на несколько, не забудь добавить необходимый контекст для независимого поиска по базе.

Результат выведи списком запросов после ключевого слова ЗАПРОСЫ: ["запрос1", "запрос2"...]
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{context}

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""


@chain
def prepare_context(retrieval_result):
    docs = retrieval_result['retrieved_docs']
    context = "\n".join([f"Параграф {i + 1}: {doc.page_content}" for i, doc in enumerate(docs)])
    return context


class HydeOutputParser(BaseOutputParser[list[str]]):
    def parse(self, text: str) -> str:
        """Parse by splitting."""
        result = text.split('\nАбзац:\n')[1]
        return result


class PlannerOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        """Parse by splitting."""
        try:
            result = ast.literal_eval(text.split('\nЗАПРОСЫ:')[1].strip().split(']')[0] + ']')
        except:
            raise OutputParserException('Failed to parse')
        return result


class BatchQueryRetriever:
    def __init__(self, retriever: VectorStoreRetriever):
        self._retriever = retriever

    def _remove_duplicates(self, docs: List[Document]) -> List[Document]:
        return list({doc.page_content: doc for doc in docs}.values())

    def do_batch(self, queries: List[str]) -> List[Document]:
        result = self._retriever.batch(queries)
        result = self._remove_duplicates([doc for docs in result for doc in docs])
        return result


class RAGPipeline:
    def __init__(self,
                 llm: HuggingFacePipeline,
                 store: VectorStore,
                 hyde: bool = False,
                 do_planning: bool = False,
                 return_intermediate_results: bool = False):
        self._llm = llm
        self._store = store
        self._hyde = hyde
        self._do_planning = do_planning
        self._return_intermediate_results = return_intermediate_results

    def build_chain(self, retrieval_only: bool = False) -> Runnable:
        retriever = self._store.as_retriever(search_kwargs={"k": 3})

        if self._hyde:
            hyde_prompt = PromptTemplate(template=hyde_prompt_template, input_variables=["question"])
            hyde_chain = (
                    hyde_prompt
                    | self._llm
                    | HydeOutputParser()
            )
            retrieving_chain = (
                    hyde_chain
                    | RunnableParallel(retrieved_docs=retriever, hyde_doc=RunnablePassthrough())
            )

        elif self._do_planning:
            planner_prompt = PromptTemplate(input_variables=["context"], template=planner_prompt_template)
            planner_parser = PlannerOutputParser()
            retry_planner_parser = RetryOutputParser.from_llm(
                parser=planner_parser,
                llm=self._llm,
                prompt=PromptTemplate.from_template("{prompt}")
            )

            completion_chain = (
                    planner_prompt
                    | self._llm
            )
            planner_chain = (
                    RunnableParallel(completion=completion_chain, prompt_value=planner_prompt)
                    | RunnableLambda(lambda x: retry_planner_parser.parse_with_prompt(**x))
            )

            bq_retriever = BatchQueryRetriever(retriever)
            retrieving_chain = (
                    planner_chain
                    | RunnableParallel(retrieved_docs=bq_retriever.do_batch, planned_queries=RunnablePassthrough())
            )

        else:
            retrieving_chain = RunnableParallel(retrieved_docs=retriever)

        if retrieval_only:
            return retrieving_chain

        context_chain = RunnableParallel(
            context=itemgetter('retrieval_result') | prepare_context,
            question=itemgetter('question'),
            retrieval_result=itemgetter('retrieval_result')
        )

        @chain
        def generate_answer(query_components):
            retrieval_result = query_components.pop('retrieval_result')

            qa_prompt = PromptTemplate(template=answer_prompt_template, input_variables=["context", "question"])
            answer_chain = (qa_prompt | self._llm)
            answer = answer_chain.invoke(query_components)

            if self._return_intermediate_results:
                return {'response': answer, **retrieval_result}
            else:
                return {'response': answer}

        rag_chain = (
                RunnableParallel(retrieval_result=retrieving_chain, question=RunnablePassthrough())
                | context_chain
                | generate_answer
        )

        return rag_chain
