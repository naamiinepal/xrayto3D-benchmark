## Custom Loss function Notes

## Normalized Gradient Cross-Correlation (1 - NGCC)
In our real dataset sample, this loss can get as low as 0.5.
Gradients of AP and LAT images are correlated with Projections of Predicted Segmentations as shown below.

![ap](ap_after_loading.png)
![lat](lat_after_loading.png)
![g_x_ap](g_x_ap.png)
![g_y_ap](g_y_ap.png)
![g_x_seg](g_x_seg.png)
![g_y_seg](g_y_seg.png)
![ngcc_loss_plot](ngcc_loss_plot.png)

The NGCC seems to be well-correlated with other losses like cross-entropy and Dice.