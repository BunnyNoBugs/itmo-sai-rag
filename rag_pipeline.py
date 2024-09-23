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
    context = "\n\n".join(doc.page_content for doc in docs)
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
                 return_retrieved_docs: bool = False):
        self._llm = llm
        self._store = store
        self._hyde = hyde
        self._do_planning = do_planning
        self._return_retrieved_docs = return_retrieved_docs

    def build_chain(self) -> Runnable:
        retriever = self._store.as_retriever(search_kwargs={"k": 3})

        if self._hyde:
            hyde_prompt = PromptTemplate(template=hyde_prompt_template, input_variables=["question"])
            hyde_chain = (
                    hyde_prompt
                    | self._llm
                    | HydeOutputParser()
            )
            retrieving_chain = RunnableParallel(retrieved_docs=hyde_chain | retriever)
        else:
            retrieving_chain = RunnableParallel(retrieved_docs=retriever)

        context_chain = RunnableParallel(
            context=itemgetter('retrieved_docs') | prepare_context,
            question=itemgetter('question'),
            retrieved_docs=itemgetter('retrieved_docs')
        )

        @chain
        def generate_answer(query_components):
            # because itemgetter in a chain returns a nested dict
            retrieved_docs = query_components.pop('retrieved_docs')['retrieved_docs']

            qa_prompt = PromptTemplate(template=answer_prompt_template, input_variables=["context", "question"])
            answer_chain = (qa_prompt | self._llm)
            answer = answer_chain.invoke(query_components)

            if self._return_retrieved_docs:
                return {'response': answer, 'retrieved_docs': retrieved_docs}
            else:
                return {'response': answer}

        rag_chain = (
                RunnableParallel(retrieved_docs=retrieving_chain, question=RunnablePassthrough())
                | context_chain
                | generate_answer
        )

        return rag_chain
