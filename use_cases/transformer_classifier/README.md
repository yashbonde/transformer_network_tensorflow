# Transformer Stripped

This is a special use case where the model was used to rank passages. The network input was `input_query` and the output sentence was `input_passages`.
The resulting output of the network was a sigmoid rank given to it in `[-1, 1]`.
The complete code for this is available [here](https://github.com/yashbonde/Babylon) that shows how the sentence numpy dumps were obtained and code for it.

## Files

Following files are uploaded here:

1. `network.py`: This has the comeplete neural network
2. `net_test.py`: This has code to check all the functionality of the network. NOTE: this is an old file, if there is some problem raise an issue in this repo.
3. `train.py`: This has code to run the network and train the model from scratch
4. `eval.py`: This has code to run the trained network and store the results in a `.tsv` file.
5. `utils.py`: Util functions.

## Usage

Files `train.py` and `eval.py` are ports to train the model. They have a variety of functions and inputs that can be seen through:

```
$ python3 train.py --help
$ python3 eval.py --help
```
