from models import CompleteOutput, Images, Videos
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms.gpt4all import GPT4All
from langchain.schema.runnable import RunnablePassthrough

class LocalLLM:
    
    def __init__(self) -> None:

        template = """[INST]Question: {question}

        Answer: Let's think step by step. If possible use a philosophical approach and tone[/INST]"""
        self.prompt = PromptTemplate.from_template(template)
        local_path = ("./models/mistral-7b-instruct-v0.1.Q4_0.gguf") 
        callbacks = [StreamingStdOutCallbackHandler()]
        self.llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True, n_threads=4, max_tokens=32000, n_predict=1024)

        self.chain = ({"question": RunnablePassthrough()} | self.prompt | self.llm)
        self.status = True

    async def execute(self, question: str) -> CompleteOutput:
        return CompleteOutput(bodyText=self.chain.invoke(question))