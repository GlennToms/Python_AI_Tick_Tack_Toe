import tensorflow as tf

print(tf.test.gpu_device_name())

if len(tf.test.gpu_device_name()) > 0:
    print(f"Default GPU Device: {tf.test.gpu_device_name()}")
else:
    print("Please install GPU version of TF")
