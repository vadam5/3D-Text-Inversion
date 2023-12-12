# Stand alone object dream fusion baseline 
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a <lego_harry_potter>" 
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a <lego_harry_potter> eating a burger"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a <lego_harry_potter> cooking a pot of spaghetti"

