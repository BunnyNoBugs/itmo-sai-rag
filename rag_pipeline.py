from langchain.llms import HuggingFacePipeline
from langchain_core.runnables.base import RunnableParallel, RunnableLambda, Runnable
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.base import BaseOutputParser
from langchain.output_parsers.retry import RetryOutputParser
from langchain_core.runnables import chain, RunnableParallel, RunnableLambda
from langchain_core.exceptions import OutputParserException
from operator import itemgetter
from langchain_core.vectorstores import VectorStoreRetriever
from typing import List

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

    def build_chain(self) -> Runnable:
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
        else:
            retrieving_chain = RunnableParallel(retrieved_docs=retriever, hyde_doc=lambda x: None)  # todo: simplify

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
