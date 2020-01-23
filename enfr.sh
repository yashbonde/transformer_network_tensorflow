python3 lang_trainer.py --filepath=./data/en.txt --spm_model=data/debates.model\
    --save_folder=./poppa_final --num_epochs=20 --log_every=2 --use_inverse_embedding=True\
    --batch_size=64 --beam_size=10

# python3 lang_trainer.py --mode=v2 --filepath=./enbu/master_joined.txt --spm_model=./enbu/enbu3.model\
#     --save_folder=./models/banyanv3 --num_epochs=100 --log_every=1 --use_inverse_embedding=True\
#     --beam_size=5 --batch_size=64 --num_layers=3 --num_heads=8 --embedding_dim=64\
#     --lr=0.05

# python3 lang_trainer.py --mode=v2 --filepath=./enbu/master_joined.txt --spm_model=./enbu/enbu3.model\
#     --save_folder=./models/banyanv3 --num_epochs=100 --log_every=1 --use_inverse_embedding=True\
#     --beam_size=5 --batch_size=64 --num_layers=3 --num_heads=2 --embedding_dim=64\
#     --lr=0.2

# python3 lang_trainer.py --mode=v2 --filepath=./enbu/master_joined.txt --spm_model=./enbu/enbu3.model\
#     --save_folder=./models/banyanv3 --num_epochs=100 --log_every=1 --use_inverse_embedding=True\
#     --beam_size=5 --batch_size=128 --num_layers=6 --num_heads=4 --embedding_dim=256\
#     --lr=0.0003