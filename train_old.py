import sys
import time

import fire
import torch
from torch.optim import Adam

from paragraphvec.data import LoadDataset, NCEData
from paragraphvec.loss import NegativeSampling
from paragraphvec.models import DM, DBOW
from paragraphvec.utils import save_training_state


def start(data_file_name="",
        eval_data_file_name="",
        num_noise_words=1,
        vec_dim=10,
        num_epochs=5,
        batch_size=8,
        lr=1e-3,
        model_ver='dbow',
        context_size=0,
        vec_combine_method='concat',
        save_all=False,
        generate_plot=True,
        max_generated_batches=5,
        num_workers=-1
    ):
    """Trains a new model. The latest checkpoint and the best performing
    model are saved in the *models* directory.

    Parameters
    ----------
    data_file_name: str
        Name of a file in the *data* directory.

    eval_data_file_name: str
        Name of a file in the *data* directory to use for eval.

    model_ver: str, one of ('dm', 'dbow'), default='dbow'
        Version of the model as proposed by Q. V. Le et al., Distributed
        Representations of Sentences and Documents. 'dbow' stands for
        Distributed Bag Of Words, 'dm' stands for Distributed Memory.

    vec_combine_method: str, one of ('sum', 'concat'), default='sum'
        Method for combining paragraph and word vectors when model_ver='dm'.
        Currently only the 'sum' operation is implemented.

    context_size: int, default=0
        Half the size of a neighbourhood of target words when model_ver='dm'
        (i.e. how many words left and right are regarded as context). When
        model_ver='dm' context_size has to greater than 0, when
        model_ver='dbow' context_size has to be 0.

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

    max_generated_batches: int, default=5
        Maximum number of pre-generated batches.

    num_workers: int, default=-1
        Number of batch generator jobs to run in parallel. If value is set
        to -1 number of machine cores are used.
    """

    if model_ver not in ('dm', 'dbow'):
        raise ValueError("Invalid version of the model")

    model_ver_is_dbow = (model_ver == 'dbow')
    if model_ver_is_dbow and context_size != 0:
        raise ValueError("Context size has to be zero when using dbow")
    
    if not model_ver_is_dbow:
        if vec_combine_method not in ('sum', 'concat'):
            raise ValueError("Invalid method for combining paragraph and word vectors when using dm")
        if context_size <= 0:
            raise ValueError("Context size must be positive when using dm")

    dataset = LoadDataset(data_file_name)
    nce_data = NCEData(dataset, batch_size, context_size, num_noise_words, max_generated_batches, num_workers)
    nce_data.start()

    eval_dataset = LoadDataset(eval_data_file_name)
    eval_nce_data = NCEData(eval_dataset, batch_size, context_size, num_noise_words, max_generated_batches, num_workers)
    eval_nce_data.start()

    try:
        _run(data_file_name, dataset, nce_data.get_generator(), len(nce_data), nce_data.vocabulary_size(),
             eval_data_file_name, eval_dataset, eval_nce_data.get_generator(), len(eval_nce_data), eval_nce_data.vocabulary_size(),
             context_size, num_noise_words, vec_dim,
             num_epochs, batch_size, lr, model_ver, vec_combine_method,
             save_all, generate_plot, model_ver_is_dbow)
    except KeyboardInterrupt:
        nce_data.stop()


def _run(data_file_name, dataset, data_generator, num_batches, vocabulary_size,
         eval_data_file_name, eval_dataset, eval_data_generator, eval_num_batches, eval_vocabulary_size,
         context_size, num_noise_words, vec_dim, num_epochs, batch_size, lr,
         model_ver, vec_combine_method, save_all, generate_plot, model_ver_is_dbow):

    if model_ver_is_dbow:
        model = DBOW(vec_dim, num_docs=len(dataset), num_words=vocabulary_size)
    else:
        model = DM(vec_dim, num_docs=len(dataset), num_words=vocabulary_size)

    cost_func = NegativeSampling()
    optimizer = Adam(params=model.parameters(), lr=lr)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU training enabled.")
    else:
        device = torch.device("cpu")
        print("CPU training enabled.")
    
    print(f"Count of documents: {len(dataset)}.")
    print(f"Unique words (vocab): {vocabulary_size}.")
    print(f"Total words: {dataset.TotalWords}")
    print(f"Num Batches: {num_batches}\n")

    best_loss = float("inf")
    prev_model_file_path = None
    model.to(device)

    print("Training started.")
    for epoch_i in range(num_epochs):
        epoch_start_time = time.time()
        loss = []
        eval_loss = []

        for batch_i in range(num_batches):
            batch = next(data_generator)
            batch.to(device)
            print(batch.doc_ids.shape)
            print(batch.target_noise_ids.shape)

            eval_batch = next(eval_data_generator)
            eval_batch.to(device)

            if model_ver_is_dbow:
                x = model.forward(batch.doc_ids, batch.target_noise_ids)
            else:
                x = model.forward(batch.context_ids, batch.doc_ids, batch.target_noise_ids)

            # calculate loss
            x = cost_func.forward(x)
            loss.append(x.item())
            
            # backpropagation
            model.zero_grad()
            x.backward()
            optimizer.step()

            # eval
            _, el = model.infer_vec(eval_batch.target_noise_ids, cost_func.forward, num_epochs=10)
            eval_loss.append(el.item())
            
            if batch_i % 100 == 0:
                _print_progress(epoch_i, batch_i, num_batches)

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
            data_file_name, model_ver, vec_combine_method, context_size,
            num_noise_words, vec_dim, batch_size, lr, epoch_i, loss, state,
            save_all, generate_plot, is_best_loss, prev_model_file_path,
            model_ver_is_dbow
        )

        epoch_total_time = round(time.time() - epoch_start_time)
        print(" ({:d}s) - loss: {:.4f} - eval_loss: {:.4f}".format(epoch_total_time, loss, eval_loss))


def _print_progress(epoch_i:int, batch_i:int, num_batches:int):

    progress = round((batch_i + 1) / num_batches * 100)
    print(f"\rEpoch {epoch_i + 1:d} - {progress:d}%", end='', file=sys.stdout, flush=True)


if __name__ == '__main__':
    fire.Fire()
