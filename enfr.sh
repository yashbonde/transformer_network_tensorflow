python3 lang_trainer.py --filepath=./data/en.txt --spm_model=data/debates.model\
    --save_folder=./poppa_final --num_epochs=20 --log_every=2 --use_inverse_embedding=True\
    --batch_size=64 --beam_size=10