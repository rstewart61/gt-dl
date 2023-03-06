from datasets import load_dataset
from transformers import RobertaTokenizer
from datasets import ClassLabel
from transformers import AdapterTrainer, Trainer


class ClassificationFinetuning:

    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.train_dataset = None
        self.test_dataset = None
        self.validation_dataset = None
        self.labels = None
        self.model = None
        self.training_args = None
        self.metrics = None
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.labels = ClassLabel(names_file=data_dir + '/labels.txt')

    def init_config(self):
        pass

    def init_model(self):
        pass

    def init_training_args(self):
        pass

    def init_metric(self):
        pass

    def load_data(self):
        train_dataset = load_dataset('json',
                                     data_files=self.data_dir + "/train.jsonl")
        validation_dataset = load_dataset('json',
                                          data_files=self.data_dir + "/dev.jsonl")
        test_dataset = load_dataset('json',
                                    data_files=self.data_dir + "/test.jsonl")

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_dataset = validation_dataset

    def encode_batch(self, batch):
        """Encodes a batch of input data using the model tokenizer."""
        tokenized_batch = self.tokenizer(batch["text"],
                                    max_length=128, truncation=True, padding="max_length")

        tokenized_batch['labels'] = self.labels.str2int(batch['labels'])
        return tokenized_batch

    def preprocess_data(self):
        self.train_dataset.rename_column_("label", "labels")
        self.validation_dataset.rename_column_("label", "labels")
        self.test_dataset.rename_column_("label", "labels")

        self.train_dataset = self.train_dataset.map(self.encode_batch, batched=True)
        self.validation_dataset = self.validation_dataset.map(self.encode_batch, batched=True)
        self.test_dataset = self.test_dataset.map(self.encode_batch, batched=True)

        # Transform to pytorch tensors and only output the required columns
        self.train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        self.validation_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        self.test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def train_adapter(self):
        self.load_data()
        self.preprocess_data()
        self.init_config()
        self.init_model()
        self.init_training_args()
        self.init_metric()
        trainer = AdapterTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset['train'],
            eval_dataset=self.validation_dataset['train'],
            compute_metrics=self.metrics,
        )

        train_metrics = trainer.train()
        val_metrics = trainer.evaluate()

        print("Train ", train_metrics)
        print("Val ", val_metrics)

        return trainer

    def train_without_adapter(self):
        self.load_data()
        self.preprocess_data()
        self.init_config()
        self.init_model()
        self.init_training_args()
        self.init_metric()

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset['train'],
            eval_dataset=self.validation_dataset['train'],
            compute_metrics=self.metrics,
        )

        train_metrics = trainer.train()
        val_metrics = trainer.evaluate()

        print("Train ", train_metrics)
        print("Val ", val_metrics)

        return trainer

    def test_on_testset(self, trainer):
        predictions = trainer.predict(self.test_dataset["train"])

        test_metrics = self.metrics((predictions.predictions, predictions.label_ids))
        print(test_metrics)