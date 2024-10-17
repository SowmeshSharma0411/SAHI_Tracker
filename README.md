Changes Done:

* Custom yolo model trained. we saw that 50 epochs works the best. Fr now we'll stick to that
* sahi we need to integrate cz we need long detection span to get better trajectory representations
* sahi on 21 slices takes 11 min 30s for 1 min vid. We can make it 8 slices(takes around 5 min fr 1 min id)....we need to adjust the sahi params:

  slice_height = 512,
  slice_width = 512,
  overlap_height_ratio = 0.75,
  overlap_width_ratio = 0.75
* Also everything runs on cpu. We need to try our best to run it on gpu. cupy/pytorch instead of numpy
@Shivgouda look into this
