# DeepSORT
An Object tracking Algorithm on the DeepSORT algorithm &amp; a YOLOv8 Detector

## introduction

The leading approach in multiple object tracking is tracking-by-detection, which utilizes object detection techniques. Typically, object trajectories are determined through global optimization problems that process entire video batches at once. Various frameworks, such as flow network formulations and probabilistic graphical models, have been adopted for this purpose. However, these methods are not suitable for online scenarios where the identification of targets is required at each time step. 
In such cases, more traditional methods like Multiple Hypothesis Tracking (MHT) and the Joint Probabilistic Data Association Filter (JPDAF) are used, performing data association frame-by-frame. Despite the recent revival of MHT and JPDAF for tracking-by-detection, their performance is computationally complex. On the other hand, Simple Online and Realtime Tracking (SORT) offers a simpler approach using Kalman filtering and frame-by-frame data association with the Hungarian method. 

SORT demonstrates favorable performance at high frame rates, especially when paired with a state-of-the-art people detector. However, SORT has limitations with tracking through occlusions due to its association metric, which relies on low state estimation uncertainty. To address this issue, a more informed metric that combines motion and appearance information, specifically a Deep Appearance Descriptor based on convolutional neural network trained on a large-scale person re-identification dataset, is proposed. This integration improves robustness against occlusions while maintaining ease of implementation and efficiency, making it suitable for online scenarios.

![applsci-12-01319-g003](https://github.com/Mazen-Elaraby/DeepSORT/assets/99294980/a16a3960-664d-4396-bf3f-8dd7560b92fc)

## Track Handling and State Estimation

The track handling and Kalman filtering framework is mostly identical to the original formulation in SORT. The authors assume a very general tracking scenario where the camera is uncalibrated and where we have no ego-motion information available. While these circumstances pose a challenge to the filtering framework, it is the most common setup considered in recent multiple object tracking benchmarks.

The tracking scenario is defined on the eight dimensional state space: (bounding box center position, aspect ratio, height) and their respective velocities

$$(u,v,\gamma,h,\dot(u),\dot(v),\dot(\gamma),\dot(h))$$

observations of the object state:
$$(u,v,\gamma,h)$$

## Assignment Problem

A conventional way to solve the association between the predicted Kalman states and newly arrived measurements is to build an assignment problem that can be solved using the Hungarian algorithm. Into this problem formulation we integrate motion and appearance information through combination of two appropriate metrics.

To incorporate motion information we use the (squared) Mahalanobis distance between predicted Kalman states and newly arrived measurements:

$$
d^{(1)}(i, j)=\left(\boldsymbol{d}_j-\boldsymbol{y}_i\right)^{\mathrm{T}} \boldsymbol{S}_i^{-1}\left(\boldsymbol{d}_j-\boldsymbol{y}_i\right)
$$

The Mahalanobis distance takes state estimation uncertainty into account by measuring how many standard deviations the detection is away from the mean track location. Further, using this metric it is possible to exclude unlikely associations by thresholding the Mahalanobis distance at a 95% confidence interval computed from the $\chi^2$ distribution We denote this decision with an indicator:

$$
b_{i, j}^{(1)}=\mathbb{1}\left[d^{(1)}(i, j) \leq t^{(1)}\right]
$$

The Mahalanobis distance is usually effective for associating objects when there is low uncertainty in motion. However, in our problem formulation involving image space, the predicted state distribution obtained from the Kalman filtering framework only provides a rough estimate of object location. Camera motion can introduce rapid displacements in the image plane, making the Mahalanobis distance less reliable for tracking objects through occlusions. Therefore, we incorporate a second metric into the assignment problem. This additional metric involves computing an appearance descriptor for each bounding box detection and maintaining a gallery of the most recent $L_k$ associated appearance descriptors for each track. The second metric determines the smallest cosine distance between the i-th track and j-th detection in appearance space.

$$
d^{(2)}(i, j)=\min \left\lbrace 1-\boldsymbol{r}_j{ }^{\mathrm{T}} \boldsymbol{r}_k^{(i)} \mid \boldsymbol{r}_k^{(i)} \in \mathcal{R}_i\right\rbrace
$$

$$
b_{i, j}^{(2)}=\mathbb{1}\left[d^{(2)}(i, j) \leq t^{(2)}\right]
$$

In combination, both metrics complement each other by serving different aspects of the assignment problem. On the one hand, the Mahalanobis distance provides information about possible object locations based on motion that are particularly useful for short-term predictions. On the other hand, the cosine distance considers appearance information that are particularly useful to recover identities after long-term occlusions, when motion is less discriminative. To build the association problem we combine both metrics using a weighted sum

$$
c_{i, j}=\lambda d^{(1)}(i, j)+(1-\lambda) d^{(2)}(i, j)
$$

where we call an association admissible if it is within the gating region of both metric:

$$
b_{i, j}=\prod_{m=1}^2 b_{i, j}^{(m)}
$$

## Matching Cascade

In order to address the challenges posed by occlusions and uncertain object locations, The authors propose an alternative approach to solving measurement-to-track associations. Instead of solving a global assignment problem, They introduce a cascade that solves a series of subproblems. The motivation for this approach arises from the observation that when an object remains occluded for a prolonged period, the uncertainty associated with its location increases over time. As a result, the probability distribution in the state space becomes more spread out, and the observation likelihood becomes less concentrated. It is essential for the association metric to account for this spread by increasing the measurement-to-track distance. However, counterintuitively, the conventional Mahalanobis distance tends to favor larger uncertainties, as it reduces the distance in terms of standard deviations between any detection and the projected track mean. This undesired behavior can lead to increased track fragmentation and unstable tracks. To address this issue, a matching cascade is proposed that prioritizes objects that are seen more frequently, thereby incorporating our understanding of how probability spreads in the association likelihood.

<p align="center">
  <img src="https://github.com/Mazen-Elaraby/DeepSORT/assets/99294980/955ccc43-3298-462e-b2f9-5787e392b5bd" />
</p>

## Deep Appearance Descriptor

By using simple nearest neighbor queries without additional metric learning, successful application of the method requires a well-discriminating feature embedding to be trained offline, before the actual online tracking application. To this end, we employ a CNN that has been trained on a large-scale person re-identification dataset (MARS) that contains over 1,100,000 images of 1,261 pedestrians, making it well suited for deep metric learning in a people tracking context. The CNN architecture of our network is shown in The following figure. In summary, The authors employ a wide residual network with two convolutional layers followed by six residual blocks. The global feauture map of dimensionality 128 is computed in dense layer 10. A final batch and $l2$ normalization projects features onto the unit hypersphere to be compatible with our cosine appearance metric

<p align="center">
  <img src="https://github.com/Mazen-Elaraby/DeepSORT/assets/99294980/5577ffc7-59bf-44c1-9c30-0a271ce3721b" />
</p>
