import sys
import time

import fire
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from paragraphvec.data_new import LoadDataset, CustomDataset
from paragraphvec.loss import NegativeSampling
from paragraphvec.models import DM, DBOW
from paragraphvec.utils import save_training_state


def start(data_file_name="",
        eval_data_file_name="",
        num_noise_words=1,
        vec_dim=10,
        num_epochs=5,
        batch_size=32,
        lr=1e-3,
        vec_combine_method='concat',
        save_all=False,
        generate_plot=True,
        max_generated_batches=64,
    ):
    """Trains a new model. The latest checkpoint and the best performing
    model are saved in the *models* directory.

    Parameters
    ----------
    data_file_name: str
        Name of a file in the *data* directory.

    eval_data_file_name: str
        Name of a file in the *data* directory to use for eval.

    vec_combine_method: str, one of ('sum', 'concat'), default='sum'
        Method for combining paragraph and word vectors when model_ver='dm'.
        Currently only the 'sum' operation is implemented.

    num_noise_words: int
        Number of noise words to sample from the noise distribution.

    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).

    num_epochs: int
        Number of iterations to train the model (i.e. number
        of times every example is seen during training).

    batch_size: int
        Number of examples per single gradient update.

    lr: float
        Learning rate of the Adam optimizer.

    save_all: bool, default=False
        Indicates whether a checkpoint is saved after each epoch.
        If false, only the best performing model is saved.

    generate_plot: bool, default=True
        Indicates whether a diagnostic plot displaying loss value over
        epochs is generated after each epoch.

    max_generated_batches: int, default=64
        Maximum number of pre-generated batches.
    """

    dataset = LoadDataset(num_noise_words, data_file_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    eval_dataset = LoadDataset(num_noise_words, eval_data_file_name)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    _run(dataset, dataloader, eval_dataset, eval_dataloader, num_noise_words, vec_dim, num_epochs, batch_size,
         lr, vec_combine_method, save_all, generate_plot, data_file_name)


def _run(dataset: CustomDataset, dataloader: DataLoader, eval_dataset: CustomDataset, eval_dataloader: DataLoader,
         num_noise_words: int, vec_dim: int, num_epochs: int, batch_size: int, lr: float, vec_combine_method: str,
         save_all: bool, generate_plot: bool, data_file_name: str):

    model = DBOW(vec_dim, num_docs=len(dataset), num_words=dataset.VocabSize)

    cost_func = NegativeSampling()
    optimizer = Adam(params=model.parameters(), lr=lr)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU training enabled.")
    else:
        device = torch.device("cpu")
        print("CPU training enabled.")
    
    print(f"Count of documents: {len(dataset)}.")
    print(f"Unique words (vocab): {dataset.VocabSize}.")
    print(f"Total words: {dataset.TotalWords}")

    best_loss = float("inf")
    prev_model_file_path = None
    model.to(device)

    print("Training started.")
    for epoch_i in range(num_epochs):
        epoch_start_time = time.time()
        loss = []
        eval_loss = []

        for batch_i, batch in enumerate(dataloader):
            batch['doc'].squeeze_(1)

            x = model.forward(batch['doc'], batch['tgt'])

            # calculate loss
            x = cost_func.forward(x)
            loss.append(x.item())
            
            # backpropagation
            model.zero_grad()
            x.backward()
            optimizer.step()
            
            if batch_i % 100 == 0:
                _print_progress(epoch_i, batch_i)

        # eval
        eval_loss = []
        for eval_batch in eval_dataloader:
            _, el = model.infer_vec(eval_batch['tgt'], cost_func.forward, num_epochs=100)
            eval_loss.append(el.item())

        # end of epoch
        loss = torch.mean(torch.FloatTensor(loss))
        eval_loss = torch.mean(torch.FloatTensor(eval_loss))
        is_best_loss = loss < best_loss
        best_loss = min(loss, best_loss)

        state = {
            'epoch': epoch_i + 1,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer_state_dict': optimizer.state_dict()
        }

        prev_model_file_path = save_training_state(
            data_file_name, vec_combine_method, num_noise_words, vec_dim,
            batch_size, lr, epoch_i, loss, state, save_all, generate_plot,
            is_best_loss, prev_model_file_path)

        epoch_total_time = round(time.time() - epoch_start_time)
        print(" ({:d}s) - loss: {:.4f} - eval_loss: {:.4f}".format(epoch_total_time, loss, eval_loss))


def _print_progress(epoch_i:int, batch_i:int):
    progress = (batch_i + 1)
    print(f"\rEpoch {epoch_i + 1:d} - batch {progress:d}", end='', file=sys.stdout, flush=True)


if __name__ == '__main__':
    fire.Fire()
