# :fire: InterLight: Leveraging Intrinsic Image Priors for Low-Light Enhancement (Under Review)



This is the official PyTorch codes for the paper:

>**InterLight: Leveraging Intrinsic Image Priors for Low-Light Enhancement**<br> Ziqi Wang<sup>1</sup>, [Xu Zhang<sup>1📧</sup>](https://house-yuyu.github.io/), [Laibin Chang<sup>1</sup>](https://scholar.google.com.hk/citations?user=1l8X8PgAAAAJ&hl=zh-CN&oi=ao), [Shi Chen<sup>2</sup>](https://scholar.google.com.hk/citations?user=4pj8flsAAAAJ&hl=zh-CN&oi=ao), [Jiaqi Ma<sup>3</sup>](https://leonmakise.github.io/), [Huan Zhang<sup>4</sup>](https://scholar.google.com.hk/citations?user=bJjd_kMAAAAJ&hl=zh-CN)<br>
> <sup>1</sup>Wuhan University <sup>2</sup>University of Macau <sup>3</sup>Mohamed bin Zayed University of Artificial
 Intelligence <sup>4</sup>Guangdong University of Technology<br>
> <sup>📧</sup> Corresponding author

## 🧠 Overview
we propose **InterLight**, a novel framework that systematically excavates and operationalizes intrinsic illumination priors for LLIE.

![teaser_img](assets/model.png)

**Key Highlights:**
- 💡 **Intrinsic-Consistent Data Expansion:** simulates sensor-level illumination responses while preserving structural fidelity.
- 🧩 **Adaptive Degradation Prior Generation:** extracts sample-specific degradation prior through a learnable degradation dictionary.
- ⚙️ **Luminance-Gated Intrinsic Memory:** retrieves learned structural and textural patterns to compensate for information loss.
## ⚡ Start

### prepare dataset

- [LOLv1](https://daooshee.github.io/BMVC2018website/)
- [LOLv2](https://pan.baidu.com/s/17KTa-6GUUW22Q49D5DhhWw?pwd=yixu) (code: `yixu`) and  [One Drive](https://1drv.ms/u/c/2985db836826d183/EYPRJmiD24UggCmCAQAAAAABEbg62rx0FG21FwLQq0jzLg?e=Im12UA) (code: `yixu`) 
- [SICE](https://pan.baidu.com/s/13ghnpTBfDli3mAzE3vnwHg?pwd=yixu) (code: `yixu`) and [One Drive](https://1drv.ms/u/s!AoPRJmiD24UphAlaTIekdMLwLZnA?e=WxrfOa) (code: `yixu`)
- [Sony-Total-Dark(SID)](https://pan.baidu.com/s/1mpbwVscbAfQJtkrrzBzJng?pwd=yixu) (code: `yixu`) and [One Drive](https://1drv.ms/u/s!AoPRJmiD24UphAie9l0DuMN20PB7?e=Zc5DcA) (code: `yixu`)
- [LSRW-Huawei](https://github.com/JianghaiSCU/R2RNet)
## :postbox: Contact

If you have any questions, please feel free to reach us out at <a href="zhangx0802@whu.edu.cn">zhangx0802@whu.edu.cn</a>.

