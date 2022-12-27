# PicT
This repo is the official implementation of ["PicT: A Slim Weakly Supervised Vision Transformer for Pavement
Distress Classification"]() based on Pytorch.

For more details of the pavement dataset CQU-BPDD used in paper, please refer to [CQU-BPDD](https://dearcaat.github.io/CQU-BPDD/).
 (Note: CQU-BPDD can be only used in the uncommercial case and is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).)
 
 For more details of this task, see [Pavement Distress Classification](https://github.com/DearCaat/Pavement-Distress-Classification).

## Usage
These examples are in the **I-REC** setting. For other settings, please change the config file.
### Train Swin-S
```shell
python3 main.py --data-path=$DATA_PATH --output=$OUTPUT_PATH --project=pict --cfg ../configs/baseline/swin_small_1rec.yaml --title=swin_s
```

### Train PicT
```shell
# PicT uses the pretrained swin_s weight to init the teacher model
python3 main.py --data-path=$DATA_PATH --output=$OUTPUT_PATH --project=pict --cfg ../configs/baseline/swin_small_1rec.yaml ../configs/pict_1rec.yaml --title=pict --opt PICT.TEACHER_INIT $PRETRAINED_WEIGHT_PATH
```
