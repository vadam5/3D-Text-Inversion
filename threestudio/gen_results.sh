# Stand alone object dream fusion baseline
#python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <war-hammer-robot>"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <crochet-octopus>" 
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <lego_harry_potter>" 

# Simple action dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <crochet-octopus> eating a burger"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <lego_harry_potter> eating a burger"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <war-hammer-robot> eating a burger"


# Simple action dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <crochet-octopus> cooking a pot of spaghetti"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <lego_harry_potter> cooking a pot of spaghetti"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <war-hammer-robot> cooking a pot of spaghetti"

# Complex scene dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <crochet-octopus> sitting on top of a basket of colorful macaroons"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <lego_harry_potter> sitting on top of a basket of colorful macaroons"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <war-hammer-robot> sitting on top of a basket of colorful macaroons"

# # Simple action dream fusion baseline
# python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <crochet-octopus> ballet dancing"
# python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <lego_harry_potter> ballet dancing"
# python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <war-hammer-robot> ballet dancing"

# Same pair simple action dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="two <crochet-octopus>s cooking a pot of spaghetti together"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="two <lego_harry_potter>s cooking a pot of spaghetti together"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="two <war-hammer-robot>s cooking a pot of spaghetti together"

# # Same pair simple action dream fusion baseline
# python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="two <crochet-octopus>s ballet dancing together"
# python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="two <lego_harry_potter>s ballet dancing together"
# python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="two <war-hammer-robot>s ballet dancing together"

# Different pair simple action dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <crochet-octopus> and a <lego_harry_potter> cooking a pot of spaghetti together"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <lego_harry_potter> and a <crochet-octopus> cooking a pot of spaghetti together"

# Different pair simple action dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <war-hammer-robot> and a <lego_harry_potter> cooking a pot of spaghetti together"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <lego_harry_potter> and a <war-hammer-robot> cooking a pot of spaghetti together"

# Different pair simple action dream fusion baseline
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <crochet-octopus> and a <war-hammer-robot> cooking a pot of spaghetti together"
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 trainer.max_steps=10000 trainer.val_check_interval=200 system.background.random_aug=true system.prompt_processor.prompt="a <war-hammer-robot> and a <crochet-octopus> cooking a pot of spaghetti together"



