# BasicIRSTD

**BasicIRSTD is a PyTorch-based open-source and easy-to-use toolbox for infrared small target detction (IRSTD). This toolbox 
introduces a simple pipeline to train/test your methods, and builds a benchmark to comprehensively evaluate the performance of existing methods.
Our BasicIRSTD can help researchers to get access to infrared small target detction quickly, and facilitates the development of novel methods. Welcome to contribute your own methods to the benchmark.**

**Note: This repository will be updated on a regular basis. Please stay tuned!**

<br>

## Contributions
* **We provide a PyTorch-based open-source and easy-to-use toolbox for IRSTD.**
* **We re-implement a number of existing methods on the unified datasets, and develop a benchmark for performance evaluation.**
* **We share the codes, models and results of existing methods to help researchers better get access to this area.**

<br>

## News & Updates
* **Apirl 4, 2022: Pulic The BasicIRSTD ToolBox.**
<br><br>

## Requirements
- **Python 3**
- **pytorch 1.2.0 or higher**
- **numpy, PIL**
<br><br>

## Datasets
* **NUST-SIRST** &nbsp; [[download]](https://github.com/wanghuanphd/MDvsFA_cGAN) &nbsp; [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Miss_Detection_vs._False_Alarm_Adversarial_Learning_for_Small_Object_ICCV_2019_paper.pdf)
* **NUAA-SIRST** &nbsp; [[download]](https://github.com/YimianDai/sirst) &nbsp; [[paper]](https://arxiv.org/pdf/2009.14530.pdf)
* **NUDT-SIRST** &nbsp; [[download]](https://github.com/YeRen123455/Infrared-Small-Target-Detection) &nbsp; [[paper]](https://ieeexplore.ieee.org/abstract/document/9864119)
* **IRSTD-1K** &nbsp; [[download dir]](https://github.com/RuiZhang97/ISNet) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9880295)
* **NUDT-SIRST-Sea** &nbsp; [[download]](https://github.com/TianhaoWu16/Multi-level-TransUNet-for-Space-based-Infrared-Tiny-ship-Detection) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/10011449/)
* **IRDST** &nbsp; [[download]](https://github.com/sun11999/RDIAN) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/10011452/)

**We used the NUAA-SIRST, NUDT-SIRST, IRSTD-1K for both training and test. 
Please first download our datasets via [Baidu Drive](https://pan.baidu.com/s/1yjSOWiViTeEswEpCKGXq3w?pwd=1113) (key:1113) and [Google Drive](https://drive.google.com/file/d/1LscYoPnqtE32qxv5v_dB4iOF4dW3bxL2/view?usp=sharing), and place the 3 datasets to the folder `./datasets/`. More results will be released soon!** 

* **Our project has the following structure:**
  ```
  ├──./datasets/
  │    ├── NUAA-SIRST
  │    │    ├── images
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_NUAA-SIRST.txt
  │    │    │    ├── test_NUAA-SIRST.txt
  │    ├── NUDT-SIRST
  │    │    ├── images
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_NUDT-SIRST.txt
  │    │    │    ├── test_NUDT-SIRST.txt
  │    ├── ...  
  ```
<br>

## Commands for Training
* **Run **`train.py`** to perform network training. Example for training [model_name] on [dataset_name] datasets:**
  ```
  $ python train.py --model_name [ACM, ALCNet] --dataset_name [NUAA-SIRST, NUDT-SIRST] --batch_size 16
  ```
* **Checkpoints and Logs will be saved to **`./log/`**, and the **`./log/`** has the following structure:**
  ```
  ├──./log/
  │    ├── [dataset_name]
  │    │    ├── [model_name]_eopch400.pth.tar
  ```

<br>

## Commands for Test
* **Run **`test.py`** to perform network inference. Example for test [model_name] on [dataset_name] datasets:**
  ```
  $ python test.py --model_name [ACM, ALCNet] --dataset_name [NUAA-SIRST, NUDT-SIRST] 
  ```
  
* **The PA/mIoU and PD/FA values of each dataset will be saved to** **`./test_[current time].txt`**<br>
* **Network preditions will be saved to** **`./results/`** **that has the following structure**:
  ```
  ├──./results/
  │    ├── [dataset_name]
  │    │   ├── [model_name]
  │    │   │    ├── XDU0.png
  │    │   │    ├── XDU1.png
  │    │   │    ├── ...
  │    │   │    ├── XDU20.png
  ```
<br>

## Commands for Evaluate on your own results
* **Please first put your results on** **`./results/`** **that has the following structure:**
  ```
  ├──./results/
  │    ├── [dataset_name]
  │    │   ├── [method_name]
  │    │   │    ├── XDU0.png
  │    │   │    ├── XDU1.png
  │    │   │    ├── ...
  │    │   │    ├── XDU20.png
  ```
* **Run **`evaluate.py`** for direct eevaluation. Example for evaluate [method_name] on [dataset_name] datasets:**
  ```
  $ python evaluate.py --method_name [ACM, ALCNet] --dataset_name [NUAA-SIRST, NUDT-SIRST] 
  ```
* **The PA/mIoU and PD/FA values of each dataset will be saved to** **`./eval_[current time].txt`**<br><br>

## Commands for parameters/FLOPs calculation
* **Run **`cal_params.py`** for parameters and FLOPs calculation. Examples:**
  ```
  $ python cal_params.py --method_name [ACM, ALCNet]
  ```
* **The parameters and FLOPs of each method will be saved to** **`./params_[current time].txt`**<br><br>

## Benchmark

**We benchmark several methods on the above datasets. mIoU, PD and FA metrics under threshold=0.5 are used for quantitative evaluation.**

**Note: A detailed review of existing IRSTD methods can be referred to [Tianfang-Zhang/awesome-infrared-small-targets](https://github.com/Tianfang-Zhang/awesome-infrared-small-targets).** 

### **mIoU/PD/FA values achieved by different methods:**
<br>

<table class=MsoTableGrid border=0 cellspacing=0 cellpadding=0
 style='border-collapse:collapse;border:none;mso-yfti-tbllook:1184;mso-padding-alt:
 0cm 5.4pt 0cm 5.4pt;mso-border-insideh:none;mso-border-insidev:none'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes;height:14.2pt'>
  <td width=83 rowspan=2 style='width:41.65pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .25pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt'>
  <p class=MsoNormal align=center style='text-align:center;line-height:12.0pt;
  mso-pagination:widow-orphan'><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>Methods<o:p></o:p></span></b></p>
  </td>
  <td width=73 rowspan=2 style='width:37.2pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .25pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt'>
  <p class=MsoNormal align=center style='text-align:center;line-height:12.0pt'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:11.0pt;
  mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;mso-fareast-font-family:
  等线;color:black'>#Params<o:p></o:p></span></b></p>
  </td>
  <td width=67 rowspan=2 style='width:34.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .25pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt'>
  <p class=MsoNormal align=center style='text-align:center;line-height:12.0pt'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:11.0pt;
  mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;mso-fareast-font-family:
  等线;color:black'>FLOPs<o:p></o:p></span></b></p>
  </td>
  <td width=196 colspan=3 style='width:100.65pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt'>
  <p class=MsoNormal align=center style='text-align:center;line-height:12.0pt'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:11.0pt;
  mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;mso-fareast-font-family:
  等线;color:black'>NUAA-SIRST<o:p></o:p></span></b></p>
  </td>
  <td width=196 colspan=3 style='width:100.65pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt'>
  <p class=MsoNormal align=center style='text-align:center;line-height:12.0pt'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:11.0pt;
  mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;mso-fareast-font-family:
  等线;color:black'>NUDT-SIRST<o:p></o:p></span></b></p>
  </td>
  <td width=196 colspan=3 style='width:100.65pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt'>
  <p class=MsoNormal align=center style='text-align:center;line-height:12.0pt'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:11.0pt;
  mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;mso-fareast-font-family:
  等线;color:black'>IRSTD-1K<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1;height:14.2pt'>
  <td width=57 style='width:30.0pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .25pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt'>
  <p class=MsoNormal align=center style='text-align:center;line-height:12.0pt'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:11.0pt;
  mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;mso-fareast-font-family:
  等线;color:black'>IoU<o:p></o:p></span></b></p>
  </td>
  <td width=57 style='width:30.0pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .25pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt'>
  <p class=MsoNormal align=center style='text-align:center;line-height:12.0pt'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:11.0pt;
  mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;mso-fareast-font-family:
  等线;color:black'>Pd<o:p></o:p></span></b></p>
  </td>
  <td width=81 style='width:40.65pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .25pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt'>
  <p class=MsoNormal align=center style='text-align:center;line-height:12.0pt'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:11.0pt;
  mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;mso-fareast-font-family:
  等线;color:black'>Fa<o:p></o:p></span></b></p>
  </td>
  <td width=57 style='width:30.0pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .25pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt'>
  <p class=MsoNormal align=center style='text-align:center;line-height:12.0pt'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:11.0pt;
  mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;mso-fareast-font-family:
  等线;color:black'>IoU<o:p></o:p></span></b></p>
  </td>
  <td width=57 style='width:30.0pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .25pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt'>
  <p class=MsoNormal align=center style='text-align:center;line-height:12.0pt'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:11.0pt;
  mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;mso-fareast-font-family:
  等线;color:black'>Pd<o:p></o:p></span></b></p>
  </td>
  <td width=81 style='width:40.65pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .25pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt'>
  <p class=MsoNormal align=center style='text-align:center;line-height:12.0pt'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:11.0pt;
  mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;mso-fareast-font-family:
  等线;color:black'>Fa<o:p></o:p></span></b></p>
  </td>
  <td width=57 style='width:30.0pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .25pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt'>
  <p class=MsoNormal align=center style='text-align:center;line-height:12.0pt'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:11.0pt;
  mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;mso-fareast-font-family:
  等线;color:black'>IoU<o:p></o:p></span></b></p>
  </td>
  <td width=57 style='width:30.0pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .25pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt'>
  <p class=MsoNormal align=center style='text-align:center;line-height:12.0pt'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:11.0pt;
  mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;mso-fareast-font-family:
  等线;color:black'>Pd<o:p></o:p></span></b></p>
  </td>
  <td width=81 style='width:40.65pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .25pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:14.2pt'>
  <p class=MsoNormal align=center style='text-align:center;line-height:12.0pt'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:11.0pt;
  mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;mso-fareast-font-family:
  等线;color:black'>Fa<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2;height:14.2pt'>
  <td width=83 style='width:41.65pt;border:none;mso-border-top-alt:solid windowtext .25pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:14.25pt;mso-pagination:widow-orphan;
  background:whitesmoke'><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>Top-Hat</span></b><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:11.0pt;
  mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;mso-fareast-font-family:
  宋体;color:#333333;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;border:none;mso-border-top-alt:solid windowtext .25pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;border:none;mso-border-top-alt:solid windowtext .25pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;border:none;mso-border-top-alt:solid windowtext .25pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>7.142 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;border:none;mso-border-top-alt:solid windowtext .25pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>79.841 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;border:none;mso-border-top-alt:solid windowtext .25pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>1012.003 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;border:none;mso-border-top-alt:solid windowtext .25pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>20.724 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;border:none;mso-border-top-alt:solid windowtext .25pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>78.408 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;border:none;mso-border-top-alt:solid windowtext .25pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>166.704 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;border:none;mso-border-top-alt:solid windowtext .25pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>10.062 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;border:none;mso-border-top-alt:solid windowtext .25pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>75.108 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;border:none;mso-border-top-alt:solid windowtext .25pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>1432.003 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>Max-Median<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>1.168 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>30.196 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>55.332 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>4.201 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>58.413 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>36.888 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>7.003 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>65.213 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>59.731 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:4;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>RLCM<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>21.022 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>80.612 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>199.154 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>15.139 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>66.348 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>162.996 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>14.623 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>65.658 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>17.949 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:5;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>WSLCM<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>1.021 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>80.987 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>45846.164 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>0.848 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>74.574 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>52391.633 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>0.989 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>70.026 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>15027.084 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:6;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>TLLCM<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>11.034 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>79.473 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>7.268 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>7.059 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>62.014 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>46.118 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>5.357 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>63.966 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>4.928 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:7;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>MSLCM<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>11.557 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>78.332 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>8.374 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>6.646 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>56.827 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>25.619 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>5.346 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>59.932 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>5.410 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:8;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>MSPCM<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>12.837 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>83.271 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>17.773 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>5.859 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>55.866 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>115.961 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>7.332 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>60.270 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>15.242 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:9;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>IPI<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>25.674 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>85.551 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>11.470 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>17.758 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>74.486 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>41.230 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>27.923 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>81.374 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>16.183 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:10;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>NRAM<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>12.164 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>74.523 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>13.852 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>6.931 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>56.403 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>19.267 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>15.249 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>70.677 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>16.926 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:11;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>RIPT<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>11.048 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>79.077 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>22.612 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>29.441 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>91.850 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>344.303 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>14.106 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>77.548 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>28.310 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:12;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>PSTNN<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>22.401 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>77.953 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>29.109 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>14.848 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>66.132 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>44.170 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>24.573 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>71.988 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>35.261 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:13;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>MSLSTIPT<o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>-<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>10.302 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>82.128 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>1131.002 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>8.341 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>47.399 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>88.102 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>11.432 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>79.027 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>1524.004 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:14;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'><a
  href="https://github.com/YimianDai/open-acm"><span style='text-decoration:
  none;text-underline:none'>ACM</span></a><o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>0.398M<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>0.402G<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>67.533 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>90.494 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>40.200 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>59.391 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>91.005 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>21.303 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>56.987 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>88.889 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>85.726 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:15;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'><a
  href="https://github.com/YimianDai/open-alcnet"><span style='text-decoration:
  none;text-underline:none'>ALCNet</span></a><o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>0.427M<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>0.378G<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>69.107 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>93.536 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>35.878 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>61.232 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>94.921 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>73.651 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>61.885 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>90.236 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>28.961 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:16;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'><a
  href="https://github.com/zhanglw882/ISTDU-Net"><span style='text-decoration:
  none;text-underline:none'>ISTDU-Net</span></a><o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>2.752M<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>7.944G<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>72.473 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>95.057 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>60.849 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>76.146 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>86.667 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>35.068 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>63.775 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>87.542 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>6.737 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:17;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'><a
  href="https://github.com/YeRen123455/Infrared-Small-Target-Detection"><span
  style='text-decoration:none;text-underline:none'>DNA-Net</span></a><o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>4.697M<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>14.261G<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>73.440 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>91.635 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>43.219 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>77.061 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>97.143 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>35.504 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>63.485 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>92.593 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>39.058 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:18;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'><a
  href="https://github.com/RuiZhang97/ISNet"><span style='text-decoration:none;
  text-underline:none'>ISNet</span></a><o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>0.966M<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>30.618G<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>79.035 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>95.057 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>7.134 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>83.157 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>94.709 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>48.810 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>65.777 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>87.879 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>21.047 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:19;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'><a
  href="https://github.com/danfenghong/IEEE_TIP_UIU-Net"><span
  style='text-decoration:none;text-underline:none'>UIU-Net</span></a><o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>50.540M<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>54.426G<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>72.075 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>94.677 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>32.517 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>80.179 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>94.392 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>14.179 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>61.358 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>86.532 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>16.227 <o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:20;mso-yfti-lastrow:yes;height:14.2pt'>
  <td width=83 style='width:41.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><b style='mso-bidi-font-weight:
  normal'><span lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;
  font-family:"Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'><a
  href="https://github.com/sun11999/RDIAN"><span style='text-decoration:none;
  text-underline:none'>RDIAN</span></a><o:p></o:p></span></b></p>
  </td>
  <td width=73 style='width:37.2pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>0.217M<o:p></o:p></span></p>
  </td>
  <td width=67 style='width:34.5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt'><span lang=EN-US style='font-size:
  11.0pt;mso-bidi-font-size:13.5pt;font-family:"Segoe UI",sans-serif;
  mso-fareast-font-family:等线;color:black'>3.718G<o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>76.938 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>96.198 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>40.680 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>79.569 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>91.958 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>15.718 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>62.333 <o:p></o:p></span></p>
  </td>
  <td width=57 style='width:30.0pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>93.266 <o:p></o:p></span></p>
  </td>
  <td width=81 style='width:40.65pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.2pt'>
  <p class=MsoNormal align=center style='margin-top:7.8pt;mso-para-margin-top:
  .5gd;text-align:center;line-height:12.0pt;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;mso-bidi-font-size:13.5pt;font-family:
  "Segoe UI",sans-serif;mso-fareast-font-family:等线;color:black'>33.744 <o:p></o:p></span></p>
  </td>
 </tr>
</table>
<br>

## Recources
* **We provide the result files generated by the aforementioned methods, and researchers can download the results via [Baidu Drive](https://pan.baidu.com/s/1va-M5ECDfjDlxCatGwjESA?pwd=1113) (key:1113) and [Google Drive](https://drive.google.com/file/d/1nF1NdPflizcgFP7cAEgOg3FC8zPmVfJJ/view?usp=sharing).**
* **The pre-trained models of the aforementioned methods can be downlaoded via [Baidu Drive](https://pan.baidu.com/s/1WL1Bt7x4rFFuFH90X3C6nA?pwd=1113) (key:1113) and [Google Drive](https://drive.google.com/file/d/1w52EmNFvQ6H3TNDzxUiNmFgP95AFTn4F/view?usp=sharing).**
<br><br>

## Acknowledgement
**We would like to thank [Boyang Li](https://github.com/YingqianWang), [Ruojing Li](https://github.com/TinaLRJ), [Tianhao Wu](https://github.com/YingqianWang) and [Ting Liu](https://github.com/LiuTing20a) for the helpful discussions and insightful suggestions regarding this repository.**
<br><br>

## Contact
**Welcome to raise issues or email to [yingxinyi18@nudt.edu.cn](yingxinyi18@nudt.edu.cn) for any question regarding our BasicIRSTD.**


