# Stand alone object dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a little orange crochet octopus" 
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a lego harry potter" 
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a war hammer robot"

# Simple action dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a little orange crochet octopus eating a burger"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a lego harry potter eating a burger"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a war hammer robot eating a burger"


# Simple action dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a little orange crochet octopus cooking a pot of spaghetti"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a lego harry potter cooking a pot of spaghetti"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a war hammer robot cooking a pot of spaghetti"

# Simple action dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a little orange crochet octopus ballet dancing"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a lego harry potter ballet dancing"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a war hammer robot ballet dancing"

# Same pair simple action dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="two little orange crochet octopuses cooking a pot of spaghetti together"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="two lego harry potters cooking a pot of spaghetti together"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="two war hammer robots cooking a pot of spaghetti together"

# Same pair simple action dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="two little orange crochet octopuses ballet dancing together"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="two lego harry potters ballet dancing together"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="two war hammer robots ballet dancing together"

# Different pair simple action dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a little orange crochet octopus and a lego harry potter cooking a pot of spaghetti together"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a lego harry potter and a little orange crochet octopus cooking a pot of spaghetti together"

# Different pair simple action dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a war hammer robot and a lego harry potter cooking a pot of spaghetti together"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a lego harry potter and a war hammer robot cooking a pot of spaghetti together"

# Different pair simple action dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a little orange crochet octopus and a war hammer robot cooking a pot of spaghetti together"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a war hammer robot and a little orange crochet octopus cooking a pot of spaghetti together"

# Complex scene dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a little orange crochet octopus sitting on top of a basket of colorful macaroons"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a lego harry potter sitting on top of a basket of colorful macaroons"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=2000 system.background.random_aug=true system.prompt_processor.prompt="a war hammer robot sitting on top of a basket of colorful macaroons"

