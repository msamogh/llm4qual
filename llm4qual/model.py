from transformers import PreTrainedModel, PretrainedConfig


class LangchainConfig(PretrainedConfig):
    pass

class LangchainModel(PreTrainedModel):

    def __init__(self, runnable):
        super().__init__(LangchainConfig())
        self.runnable = runnable

    def forward(self, text):
        return self.runnable.invoke(text)
