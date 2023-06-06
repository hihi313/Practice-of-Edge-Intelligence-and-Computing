<!-- 
原則上建議所有內容以中文撰寫，但不限制。
以下內容除標題外，其餘皆為輔助填寫說明，如有需要可自行調整項目及細節表達方式
-->

# NTUST Edge AI 111-02 期末專題報告

# 系所年級_學號_姓名

# 1. 作品名稱

深度學習特徵點提取

# 2. 摘要說明

<!-- 
這部份以100~200字，說明創作理念、硬體架構、模型選用、訓練成效、優化過程及最終結果比較。 -->

## 創作理念

### VSLAM

視覺SLAM（Visual SLAM）系統是一種使用圖像資訊(Visual)進行同時定位與地圖構建(SLAM, Simutaniously Localization And Mapping)的技術。它通過分析如輸入圖像的特徵等資訊，估計相機的/無人機/自駕車的位置並同時建立周圍環境的地圖。視覺SLAM廣泛應用於導航(尤其是無地圖資料時，可用於初步地建立地圖)、AR、VR等領域。

* 優勢：使用相對便宜且常見的相機感測器進行環境感知。
* 劣勢: 對環境變化敏感。VSLAM的性能易受傳統電腦視覺中常見的問題而影像其定位或是建圖的性能。例如，光照變化、場景結構變化或動態物體的存在可能使得特徵點提取和匹配變得困難。

### 深度學習特徵點提取

相對於傳統的手工設計特徵提取方法，深度學習特徵點提取具有以下優勢：

1. 可從大量數據中學習到更具判別性和鑑別性的特徵描述子。
2. 不依賴於先驗知識，能夠自動學習適應不同任務，更具泛化能力。

在VSLAM的應用中，傳統的視覺SLAM方法依賴於手工設計的特徵，如SIFT或ORB等，但他們容易受到環境變化的影響，導致定位和地圖構建的不穩定性。深度學習特徵點提取通過學習適應性的特徵表示，能夠更好地應對環境變化，提高視覺SLAM系統的robustness和準確性。

# 3. 系統簡介

<!-- 
至少一張結果示意影像作為代表圖示。
-->

<figure>

![SuperPoint](./src_md/SuperPoint.svg)

<figcaption>SuperPoint: 全卷積神經網絡，可同時計算 2D 興關鍵點位置和描述子</figcaption>

</figure>

## 3.1 創作發想

<!-- 
請簡單說明為何創作（如受某篇論文或某項網路作品激發、生活上常遇到問題、市場缺乏對應解決方案等等）
預期解決何種問題（分類、物件偵測、影像分割、時序預測、人臉辨識、姿態估測、模型壓縮等）及目前存在解決方案及不足的地方。
預期系統完成後可達目標（如節省人力、提高辨識精度、改善生活等等）
-->

傳統的特徵(點)提取方式容易受到光線變化的影響，如：動態的光照(閃爍的燈光、機載光源(onboard illumination))、較強的雜訊等。希望藉由深度學習的技術來讓所提取的特徵在光照變化的環境下更加穩定，並減少誤匹配的情況發生。

期望能藉由深度學習特徵點提取模型(SuperPoint)搭配之後的 SLAM 演算法研究，使的整體 VSLAM 演算法的表現、robustness 能夠比以往的演算法更佳。

## 3.2 軟硬體架構

<!-- 
說明使用硬體（如筆電、網路攝影機、麥克風、樹莓派、Jetson Nano、Arduino Nano 33 BLE Sense及其它各種輸入、輸出裝置或通訊界面等等）、連接方式及軟體階層說明等。
-->

Trainging/testing Hardware:

|Hardware type|Model/Spec|
|---|---|
|Memory|32GB|
|CPU|i7-11700 @ 2.5GHz|
|GPU|NVIDIA GeForce RTX 3090| 

Testing hardware 2:
* 主要用於測試執行時間

|Hardware type|Model/Spec|
|---|---|
|Memory|8GB/2GB swap (in container)|
|CPU|i5-1030NG7 @ 1.1GHz|
|GPU|-| 

Software environment: 

Host: 

|Software|Version|
|---|---|
|NVIDIA GPU Driver(`nvidia.ko`)|470.182.03|
|CUDA Driver(`libcuda.so`)|470.182.03|
|NVIDIA Container toolkit(`nvidia-ctk`)|1.13.1| 
|Docker| 20.10.21|


Container: 
* Based on `nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04`
* [Dockerfile](https://github.com/hihi313/Practice-of-Edge-Intelligence-and-Computing/blob/master/docker/Dockerfile)

|Software|Dependencies|
|---|---|
|apt|[apt_packages.text](https://github.com/hihi313/Practice-of-Edge-Intelligence-and-Computing/blob/master/docker/apt_packages.txt)|
|pip|[requirements](https://github.com/hihi313/Practice-of-Edge-Intelligence-and-Computing/blob/master/docker/requirements.txt)|

## 3.3 工作原理及流程

<!-- 
簡述輸入及輸出架構（如使用網路攝影機作為輸入，經過XXX硬體計算後，在螢幕或其它輸出裝置顯示結果），最好能給出一張簡單流程圖。
-->


```mermaid
%%{init: { 'securityLevel': 'antiscript'} }%%
flowchart LR
    subgraph prev["t-1"]
        direction LR
        img_p[<img src='https://github.com/hihi313/Practice-of-Edge-Intelligence-and-Computing/blob/master/src_md/tr9_000000000074.png?raw=true' width='100' /> Previous frame]--> SP_p["SuperPoint"]
        SP_p --> desc_p["Descriptor"]
        SP_p --> kp_p["Keypoint"]
    end
    subgraph cur["t"]
        direction LR
        img[<img src='https://github.com/hihi313/Practice-of-Edge-Intelligence-and-Computing/blob/master/src_md/tr9_000000000074.png?raw=true'  width='100' /> Current frame]--> SP["SuperPoint"]
        SP --> desc["Descriptor"]
        SP --> kp["Keypoint"]
    end
    desc_p --> match["OpenCV match"]
    desc --> match
    kp_p --> imshow
    match -- match info --> imshow["imshow()"]
    kp --> imshow 
```

測試方式為: 在一個迴圈內，對目前的輸入圖像輸入至 SuperPoint 深度學習特徵提取網路內，經由網路(和後處理後)提取特徵點的位置以及其對應描述子。經過轉換後直接使用 OpenCV 的特徵匹配方法，以描述子 L2 距離取最接近的作為匹配的點。最後再經由 OpenCV 將匹配的結果畫在當前幀以及前一幀上。

<figure>

![SuperPoint testing](https://github.com/hihi313/Practice-of-Edge-Intelligence-and-Computing/blob/master/src_md/superpoint_tr924_202208031659.png?raw=true)

<figcaption>SuperPoint 在自行錄製的資料集上進行推論，經 OpenCV 匹配的結果</figcaption>

</figure>

## 3.4 資料集建立方式

<!-- 
說明如何建置資料集，是採用公開或自定義資料集。
如何收集資料集及資料集數量統計說明
-->

### 資料蒐集

<figure>

![ee7](https://github.com/hihi313/Practice-of-Edge-Intelligence-and-Computing/blob/master/src_md/ee7_000000000079.png?raw=true)

<figcaption>台科 EE 7 樓錄製的資料序列</figcaption>

</figure>

<figure>

![tr9](https://github.com/hihi313/Practice-of-Edge-Intelligence-and-Computing/blob/master/src_md/tr9_000000000074.png?raw=true)

<figcaption>台科 TR 9 樓錄製的資料序列</figcaption>

</figure>


第一次訓練模型使用 MS-COCO val2017 資料集，並取其訓練後的模型做 fine-tune。Fine-tune 模型則使用自建資料集，蒐集台科 TR 9 樓 和 EE 大樓 7 樓的場景

MS-COCO val2017 資料集: 
* 因在參考的 [Github repo][1] 中使用的資料集(MS-COCO 2014)太大，考量到有自建資料集進行 fine-tune，且訓練時會做類似資料擴增(data augmentation) 的 homography transform，資料數量應足夠，所以選擇資料集大小較小的 MS-COCO val2017
* 共 5000 張圖片
* 手動分成訓練、驗證，以符合 COCO 資料集的格式。比例分別約為 8:2
    * 考量有自建資料集，以及其他的資料集，因此沒有再分出測試資料集

自建資料集: 
* 主要使用樹梅派的(魚眼)相機進行錄製，如上圖
* 手動分成訓練、驗證、測試資料集，以符合 COCO 資料集的格式。比例分別約為 8:1:1
* 訓練資料數量: 3970
* 驗證資料數量: 496
* 測試資料數量: 501


### 資料標注

<figure>

![SuperPoint training pipeline](https://media.arxiv-vanity.com/render-output/6545001/x2.png)

<figcaption>SuperPoint 自監督資料標注/訓練流程</figcaption>

</figure>

<figure>

![Homographic adaptaion](https://media.arxiv-vanity.com/render-output/6545001/x5.png)

<figcaption>Homographic adaptaion 示意圖</figcaption>

</figure>

Superpoint 論文中的自監督資料標注流程:

1. 生成虛擬/合成影像，並將已知的角/特徵點作為 ground truth 並訓練 SuperPoint 網路中的 detector
    * 不訓練 descriptor，即網路下半部，論文中稱此網路為 magicpoint
    * 生成如: 棋盤格、點、線段、2D/3D 多面體等，並在背景加入一些 noise 以提高 robustness
2. 因用虛擬影像訓練的網路在真實世界的泛化能力不足，因此透過如: Homography 等變換方式，將同張真實世界的圖經變換後所找到的所有點作為此張圖的特徵點 ground truth
    * 論文中稱為: homographic adaptation
    * 論文中稱生成的 groud truth 為: pseudo ground truth
3. 利用前步驟自監督生成的 ground truth，將真實圖像經變換(如: homography, Gaussian blur 等)，同時訓練全部網路(訓練 point detector 以及 point descriptor extractor)


<figure>

![MS-COCO val2017 #724 label](./src_md/000000000724.png)

<figcaption>使用預訓練 magicpoint 網路在 MS-COCO val2017 資料集的圖片進行 Self-supervise 產生的 label</figcaption>

</figure>


實際標注流程: 

1. 第一部份直接使用預[訓練的 magicpoint 網路](https://github.com/eric-yyjau/pytorch-superpoint/blob/master/logs/magicpoint_synth_t2/checkpoints/superPointNet_100000_checkpoint.pth.tar)
2. 使用[預訓練 magicpoint](https://github.com/eric-yyjau/pytorch-superpoint/blob/master/logs/magicpoint_synth_t2/checkpoints/superPointNet_100000_checkpoint.pth.tar) 生成自建資料集以及 MS-COCO val2017 的 pseudo ground truth

<figure>

![ee7_000000000199](./src_md/ee7_000000000199.png)

<figcaption>使用預訓練 magicpoint 網路在自行錄製資料集(EE7)標記的 label</figcaption>

</figure>

<figure>

![tr9_000000002920](./src_md/tr9_000000002920.png)

<figcaption>使用預訓練 magicpoint 網路在自行錄製資料集(TR9)標記的 label</figcaption>

</figure>

## 3.5 模型選用與訓練

<!-- 
簡述使用公開模型或自定義模型原因
是否重新訓練及相關參數設定等
-->

### 模型

<figure>

![SuperPoint architecture in paper](https://media.arxiv-vanity.com/render-output/6545001/x3.png)

<figcaption>論文中的 SuperPoint 架構圖</figcaption>

</figure>

<details>

<summary>實際模型架構，用 tensorboard 繪製 (圖大)</summary>

<figure>

![SuperPoint architecture](https://github.com/hihi313/Practice-of-Edge-Intelligence-and-Computing/blob/master/src_md/SuperPoint%20arch.png?raw=true)

<figcaption>SuperPoint 架構</figcaption>

</figure>

<figure>

![SuperPoint architecture](https://github.com/hihi313/Practice-of-Edge-Intelligence-and-Computing/blob/master/src_md/SuperPoint%20arch%20inconv.png?raw=true)

<figcaption>incvon block 架構</figcaption>

</figure>

<figure>

![SuperPoint architecture](https://github.com/hihi313/Practice-of-Edge-Intelligence-and-Computing/blob/master/src_md/SuperPoint%20arch%20down.png?raw=true)

<figcaption>down block 架構</figcaption>

</figure>

</details>

### 訓練

共分為 2 部份/步驟做訓練: 

1. 使用在 MS-COCO val2017 對模型做初步訓練
2. Fine-tune 使用自建資料集

<figure>

![SuperPoint architecture](./src_md/loss_COCO%20and%20fine-tune.png)

<figcaption>訓練 loss。灰色: 使用 COCO dataset，橘色: 使用自建資料集</figcaption>

</figure>

# 4. 實驗結果

## 4.1 測試與比較

<!-- 
說明基本實驗結果及對比其它模型或解決方案的差異。
-->

> 匹配(match): 使用 OpenCV 對特徵點描述子做匹配時，使用 L2 距離作為匹配的指標，並通過 ratio test (參考自 OpenCV 的官方教學)，篩選最近距離(的描述子) < 第二近距離 0.7 倍的點，符合此條件的點才視為匹配。隱含的意思是: 相近的點的距離要夠近，不相似的點的距離需要夠遠

> outlier: 使用 OpenCV 以 RANSAC 方式估計 fundamental 矩陣 F。設某匹配點對於上一幀的位置 x，當前幀的位置 x'，則 $l'=x'=Fx$ (對極幾何約束(epipolar constraint)，l'為極線(epipolar line))。所以若 x 在當前幀的點位置 x' 與 l' 距離過大則視為 outlier。實驗時以參數 `--f_rej_th` 設定為 5 (除非另有標示)。
> 另外，雖然場景中的動態物體也會讓誤差值較大。但若正確批配的話，理論上不論何種特徵提取方式都會受到影響。

> `f_rej_th`: 實驗過程中發現此閥值若使用預設的 3，SuperPoint 的 `outlier_prec` 很像會比較高，故設(SuperPoint 及 SIFT 的閥值皆) 為 5。推測是因為 SuperPoint 提取的特稱點的熱點圖直接使用 `max_pool` 的方式達到 NMS 的效果，所以在精度上最多只能達到像素級別。而觀察 OpenCV SIFT 的特徵點位置輸出，則其可達到 subpixel 的精度。因為目前的目標並沒有要測試其精度，目的是找出誤差較大的誤匹配，故將此閥值調高。

### 模型表現

主要以 3 個資料集進行測試與比較。除了自建資料集，選擇有動態環境光(dynamic illumination)較多的場景，希望能測試深度學習方法和傳統方法之間在此情境的差距。輸入圖片大小固定為 240\*320(H\*W)。
1. HISLAB
    * 自建資料集中的測試資料集，與訓練資料集同一個場景(EE7, TR9)。中間**場景有切換**(第199、200幀之間)，故匹配的比例驟降，以及 outlier 比例驟升。
2. OIVIO OD_015_HH_03
    * 手持設備並搭載照明系統拍攝的公開資料集，主要用於測試機載照明設備所造成的動態環境光對於特徵提取的影響
3. TartanAir ocean hard P009
    * 電腦模擬生成的資料集。水面變化會造成水下場景光線變化。電腦生成的魚類可能會造成小區域 repeated pattern，希望也能藉此測試深度學習方法和傳統方法的差異，以及深度學習方法是否能克服此問題。

因為後續希望應用在 SLAM 的系統中，所以比較的指標如下:

> SuperPoint 每個序列都有測不同的版本(torch+CPU, torch+GPU, ONNX, OpenVINO)，故有多筆資料

#### 匹配比例

* `match_prec`
* 越高越好
* (與上一幀)匹配的數量/(當前幀)特徵點數量。若當前幀無特徵點則設為0(0%)

<figure>

![HISLAB_test-match_prec](./src_md/HISLAB_test-match_prec.png "HISLAB_test-match_prec")

<figcaption>在 HISLAB 測試資料集序列上的匹配比例</figcaption>

</figure>

<figure>

![OIVIO_OD_015_HH_03-match_prec](./src_md/OIVIO_OD_015_HH_03-match_prec.png "OIVIO_OD_015_HH_03-match_prec")

<figcaption>在 OIVIO OD_015_HH_03 測試資料集序列上的匹配比例</figcaption>

</figure>

<figure>

![TartanAir_ocean_hard_P009-match_prec](./src_md/TartanAir_ocean_hard_P009-match_prec.png "TartanAir_ocean_hard_P009-match_prec")

<figcaption>在 TartanAir ocean hard P009 測試資料集序列上的匹配比例</figcaption>

</figure>

可以發現，透過批配時的 ratio test，SuperPoint 能夠將正例以及負例的 L2 距離拉開，進而使更多的點對批配成功。這也表明在論文中的 loss function 是有效的

> 論文中的 descriptor loss: $l_d\left(\mathbf{d}, \mathbf{d}^{\prime} ; s\right)=\lambda_d * s * \max \left(0, m_p-\mathbf{d}^T \mathbf{d}^{\prime}\right) +(1-s) * \max \left(0, \mathbf{d}^T \mathbf{d}^{\prime}-m_n\right)$，$\lambda_p$ 代表負例 loss 的權重

#### outlier 比例

* `outlier_prec`
* 越低越好
* outlier 數量/匹配數量。若無匹配則設為1(100%)
    <!-- * (部份實驗設錯為100，變成10000%。所以若有超過1(100%)者，實際意義仍為100%) -->

<figure>

![HISLAB_test-outlier_prec](./src_md/HISLAB_test-outlier_prec.png "HISLAB_test-outlier_prec")

<figcaption>在 HISLAB 測試資料集序列上的 outlier 比例</figcaption>

</figure>

<figure>

![OIVIO_OD_015_HH_03-outlier_prec](./src_md/OIVIO_OD_015_HH_03-outlier_prec.png "OIVIO_OD_015_HH_03-outlier_prec")

<figcaption>在 OIVIO OD_015_HH_03 測試資料集序列上的 outlier 比例，都約為 10% 以內。</figcaption>

</figure>

<figure>

![TartanAir_ocean_hard_P009-outlier_prec](./src_md/TartanAir_ocean_hard_P009-outlier_prec.png "TartanAir_ocean_hard_P009-outlier_prec")

<figcaption>在 TartanAir ocean hard P009 測試資料集序列上的 outlier 比例</figcaption>

</figure>

除了一般光線的環境下，不論是用深度學習方法或是傳統方法，outlier 的比例都差不多。但到了環境光變化較大的序列則體現深度學習方法的優勢。

#### 特徵點數量

* `num_kp`
* 一般情況下通常不會參考，因為 SuperPoint 模型架構的設計方式基本上最多每 8\*8 個 pixel 取一個特徵點，故特徵點數量相較於比較對象 SIFT 可能會比較少。但若 SuperPoint 提取特徵點數量明顯多於 SIFT ，則表示在這些場景下，SuperPoint 仍能有效提取場中的特徵。

<figure>

![HISLAB_test-num_kp](./src_md/HISLAB_test-num_kp.png "HISLAB_test-num_kp")

<figcaption>在 HISLAB 測試資料集序列上的特徵點數量</figcaption>

</figure>

<figure>

![OIVIO_OD_015_HH_03-num_kp](./src_md/OIVIO_OD_015_HH_03-num_kp.png "OIVIO_OD_015_HH_03-num_kp")

<figcaption>在 OIVIO OD_015_HH_03 測試資料集序列上的特徵點數量</figcaption>

</figure>

<figure>

![TartanAir_ocean_hard_P009-num_kp](./src_md/TartanAir_ocean_hard_P009-num_kp.png "TartanAir_ocean_hard_P009-num_kp")

<figcaption>在 TartanAir ocean hard P009 測試資料集序列上的特徵點數量</figcaption>

</figure>

在 TartanAir ocean 的環境下，傳統方式表現較差。推測是因為對比度較低，故傳統方法無法在沒有 histogram equalization 前處理下提取特徵。

## 4.2 改進與優化

<!-- 
如原來使用Nvidia GPU+PyTorch結果和經Intel OpenVINO優化後，模型壓縮大小、推論精度與速度比較。
或以不同參數找出最佳解過程。
-->

### 推論精度

由上面的比較可知，在經過 OpenVINO 優化以後的模型在以上幾個指標下的差異不大

> OpenVINO 優化指令: `mo --compress_to_fp16 --input_model "${1}" --output_dir "${OUTPUT_DIR}"`

### 推論速度

> 計算推論速度包含前處理、推論、後處理(對模型輸出 heatmap 做 NMS 等)，不包含計算以上 metric、繪圖、顯示等時間。SIFT 無後處理

<figure>

![HISLAB_test-total_time](./src_md/HISLAB_test-total_time.png "HISLAB_test-total_time")

<figcaption>在 HISLAB 測試資料集序列上的執行時間</figcaption>

</figure>

TODO: run HISLAB test on laptop

# 5. 結論

# 6. 參考資料

<!-- 
參考論文1
外部資料集及標註檔
-->

1. [Superpoint: Self-supervised interest point detection and description](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w9/html/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.html)
2. [MS-COCO val2017](https://images.cocodataset.org/zips/val2017.zip)
3. [eric-yyjau / pytorch-superpoint][1]
    * 修改並用於訓練
4. [magicleap/SuperPointPretrainedNetwork](https://github.com/magicleap/SuperPointPretrainedNetwork)
    * 參考後處理以及顯示結果 code
5. [magicleap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)
    * 參考後處理 code

# 7. 附錄

<!-- 
公開源碼連結
其它說明內容
-->

1. [https://github.com/hihi313/Practice-of-Edge-Intelligence-and-Computing](https://github.com/hihi313/Practice-of-Edge-Intelligence-and-Computing)
2. [https://github.com/hihi313/pytorch-superpoint/tree/master](https://github.com/hihi313/pytorch-superpoint/tree/master)


[1]: https://github.com/eric-yyjau/pytorch-superpoint "eric-yyjau / pytorch-superpoint"