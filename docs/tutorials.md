### Fist steps

Creating and showing a camera stream is pretty easy:

``` python
import camera_kit as ck

with ck.camera_manager('build_in_camera') as cm:
    while not ck.user_signal.stop():
        cm.render()

```
