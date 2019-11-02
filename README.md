# transformer_network_tensorflow
Tensorflow implementation of transformer network

You can learn more about the code by reading my blogs [part 1](https://medium.com/datadriveninvestor/lets-build-attention-is-all-you-need-1-2-de377cebe22) and [part 2](https://medium.com/datadriveninvestor/lets-build-attention-is-all-you-need-2-2-11d9a29219c4). This will give you the idea why and how the code is written. For following with the same and understanding the code [tutorial](./tutorial) would be a good start, however note that it is writte for easy understanding and it not the optimal implementation.

## Usage

For using the network prefer to go for [`transformer`](./transformer), it has a much more stable and scalable parts of code. In beam-search I have removed the caches as felt too clunky and difficult to use. To test the network run the command
```bash
python3 simple_trainer.py --mode=1
```

And it decodes properly as well, sample decode below:
```
seq: [[[ 3 14 14 14  8 10  8 10 10  8 10 14  0  7  7  7  7  2]
       [ 3 14 14 14  8 10  8 10 14  8 10 14  2  0  0  0  0  0]
       [ 3 14 14 14  8 10  0  7  0  8 10 14  2  0  0  0  0  0]
       [ 3 14 14 14  8 10  8 10 10  8 10 14  2  0  0  0  0  0]
       [ 3 14 14 14  8 10  0  7  0  8  2  0  0  0  0  0  0  0]
       [ 3 14 14 14  8 10  8 10  2  0  0  0  0  0  0  0  0  0]
       [ 3 14 14 14  8 10  8  2  0  0  0  0  0  0  0  0  0  0]]
```

Now that it works properly will try to use it for Reinforcement Learning Tasks such as solving sequential decision making tasks.

## Credits

There was severe limitation in understanding code from different places so I took what I found the best and easiest of their code and merged it:
* [OpenAI](https://github.com/openai/gpt-2) for just how simple and powerful their code is.
* [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor), home of the transformer. Picked up beam-decoding from here, all credits to them. As it turns out copying their code by typing eveyrthing is a great way to learn.
* [Kyubyong](https://github.com/Kyubyong) I mean who hasn't learned from and used his code.
