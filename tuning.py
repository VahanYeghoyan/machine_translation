import os
import numpy as np
import tensorflow as tf
import tensorflow_text
import random
from transformer4 import Transformer
import pickle

config = {
    'num_layers': [1],
    'd_model': np.arange(100, 146, 2),
    'batch_size': [512, 1024],
    'num_head': np.arange(4, 7),
    'dff': [256, 512, 1024, 2048],
    'dropout_rate': [0.1, 0.2],
    'max_tokens': np.arange(60, 71),
    'warmup_steps': [3000, 4000, 5000]
}

config_idx = 20
for _ in range(200):
    # increment config_idx by one at the end of iteration
    tf.keras.backend.clear_session()
    current_config = {}
    for param, seq in config.items():
        value = random.choices(seq)[0]
        current_config[param] = value

    current_config['d_model'] = int(current_config['d_model'])
    if current_config['d_model'] % current_config['num_head'] == 0:

        file_path = f'pkls/{config_idx}.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(current_config, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(current_config)

        MAX_TOKENS = int(current_config['max_tokens'])

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

        model_name = 'armenian_tokenizer'
        tokenizer = tf.saved_model.load(model_name).am

        with open("/home/vahan/Documents/machine_translation/training_data_paired", "rb") as f:
            result_dict = pickle.load(f)

        keysList = list(result_dict.keys())
        valuesList = list(result_dict.values())

        # Step 2: Convert data to tensors
        data_tensors = tf.convert_to_tensor(keysList)
        labels_tensors = tf.convert_to_tensor(valuesList)

        # Step 3: Create tuples of tensors
        data_tuples = tf.data.Dataset.from_tensor_slices((data_tensors, labels_tensors))

        # Step 4: Construct PrefetchDataset
        prefetch_dataset = data_tuples.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        dataset_size = len(prefetch_dataset)

        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size

        # Shuffle the dataset
        dataset = prefetch_dataset.shuffle(buffer_size=dataset_size, reshuffle_each_iteration=False)

        # Split the dataset
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)

        # Prefetch the datasets
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


        def prepare_batch(input, output):

            input = tokenizer.tokenize(input)  # Output is ragged.
            input = input[:, :MAX_TOKENS]  # Trim to MAX_TOKENS.
            input = input.to_tensor()  # Convert to 0-padded dense Tensor

            output = tokenizer.tokenize(output)
            output = output[:, :(MAX_TOKENS + 1)]
            decoder_inputs = output[:, :-1].to_tensor()  # Drop the [END] tokens
            targets = output[:, 1:].to_tensor()  # Drop the [START] tokens

            return (input, decoder_inputs), targets


        BUFFER_SIZE = 5000
        BATCH_SIZE = current_config['batch_size']


        def make_batches(ds):
            return (
                ds
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE)
                .map(prepare_batch, tf.data.AUTOTUNE)
                .prefetch(buffer_size=tf.data.AUTOTUNE))


        train_batches = make_batches(train_dataset)
        val_batches = make_batches(val_dataset)

        tf.keras.backend.clear_session()
        transformer = Transformer(
            num_layers=current_config['num_layers'],
            d_model=current_config['d_model'],
            num_heads=current_config['num_head'],
            dff=current_config['dff'],
            input_vocab_size=tokenizer.get_vocab_size().numpy(),
            target_vocab_size=tokenizer.get_vocab_size().numpy(),
            dropout_rate=current_config['dropout_rate'], )


        class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, d_model, warmup_steps=current_config['warmup_steps']):
                super().__init__()

                self.d_model = d_model
                self.d_model = tf.cast(self.d_model, tf.float32)

                self.warmup_steps = warmup_steps

            def __call__(self, step):
                step = tf.cast(step, dtype=tf.float32)
                arg1 = tf.math.rsqrt(step)
                arg2 = step * (self.warmup_steps ** -1.5)

                return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


        learning_rate = CustomSchedule(current_config['d_model'])

        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)


        def masked_loss(label, pred):
            mask = label != 0
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction='none')
            loss = loss_object(label, pred)

            mask = tf.cast(mask, dtype=loss.dtype)
            loss *= mask

            loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
            return loss


        def masked_accuracy(label, pred):
            pred = tf.argmax(pred, axis=2)
            label = tf.cast(label, pred.dtype)
            match = label == pred

            mask = label != 0

            match = match & mask

            match = tf.cast(match, dtype=tf.float32)
            mask = tf.cast(mask, dtype=tf.float32)
            return tf.reduce_sum(match) / tf.reduce_sum(mask)


        log_dir = f"models/d_model{current_config['d_model']}_dff{current_config['dff']}_numhead{current_config['num_head']}_numlayer{current_config['num_layers']}_bs{current_config['batch_size']}"
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        checkpoint_dir = log_dir + '/ckpt/cp.ckpt'

        with open(log_dir + "/config.txt", "w") as f:
            f.write(f"""num_layers = {current_config['num_layers']}
            d_model = {current_config['d_model']}
            dff = {current_config['dff']}
            num_heads = {current_config['num_head']}
            dropout_rate = {current_config['dropout_rate']}""")
            f.close()

        my_callbacks = transformer.get_callbacks(logdir=log_dir, checkpoint=checkpoint_dir, verbose=1)

        transformer.compile(
            loss=masked_loss,
            optimizer=optimizer,
            metrics=[masked_accuracy])

        transformer_model = transformer.build_model(input_shape=(None,), target_shape=(None,))


        # transformer_model.summary()

        class InterruptTraining(tf.keras.callbacks.Callback):
            def __init__(self, threshold, epoch_limit):
                super(InterruptTraining, self).__init__()
                self.threshold = threshold
                self.epoch_limit = epoch_limit

            def on_epoch_end(self, epoch, logs=None):
                if epoch >= self.epoch_limit and logs.get('loss') > self.threshold:
                    print(
                        f"\nEpoch {epoch + 1}: Loss {logs.get('loss')} is higher than {self.threshold}, stopping training.")
                    self.model.stop_training = True


        interrupt_callback = InterruptTraining(threshold=1, epoch_limit=4)


        class InterruptTraining2(tf.keras.callbacks.Callback):
            def __init__(self, improvement_threshold=0.01, patience=3):
                super(InterruptTraining2, self).__init__()
                self.improvement_threshold = improvement_threshold
                self.patience = patience
                self.val_accuracies = []

            def on_epoch_end(self, epoch, logs=None):
                val_accuracy = logs.get('val_masked_accuracy')
                if val_accuracy is not None:
                    self.val_accuracies.append(val_accuracy)

                    if len(self.val_accuracies) > self.patience:
                        # Only keep the last 'patience' epochs
                        self.val_accuracies.pop(0)

                        # Calculate the improvement over the first and last of these epochs
                        improvement = self.val_accuracies[-1] - self.val_accuracies[0]
                        if improvement < self.improvement_threshold:
                            print(
                                f"\nEpoch {epoch + 1}: Validation accuracy improvement {improvement:.4f} is less than {self.improvement_threshold}, stopping training.")
                            self.model.stop_training = True


        interrupt_callback2 = InterruptTraining2(improvement_threshold=0.01, patience=3)

        my_callbacks.append(interrupt_callback)
        my_callbacks.append(interrupt_callback2)

        if transformer_model.count_params() < 12000000:
            print(transformer_model.count_params())

            history = transformer.fit(train_batches,
                                      epochs=15,
                                      validation_data=val_batches,
                                      callbacks=my_callbacks)

            with open(f'history/history_{config_idx}.pkl', 'wb') as f:
                pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
            config_idx += 1
