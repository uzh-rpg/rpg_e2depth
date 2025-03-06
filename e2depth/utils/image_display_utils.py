import cv2
import numpy as np
from .timers import Timer
from .inference_utils import make_event_preview
from datetime import datetime


class Trackbar:
    def __init__(self, name, min_val, max_val, num_ticks):
        self.name = name
        self.num_ticks = num_ticks
        self.min_val, self.max_val = min_val, max_val
        self.range = self.max_val - self.min_val

    def __call__(self, val):
        return self.tick_pos_to_val(val)

    def val_to_tick_pos(self, val):
        return int(self.num_ticks * (val - self.min_val) / self.range)

    def tick_pos_to_val(self, tick_pos):
        return self.min_val + float(tick_pos) * self.range / self.num_ticks


class ImageDisplay:
    """
    Utility class to display image reconstructions
    """

    def __init__(self, options):
        self.display = options.display
        self.display_trackbars = not options.no_display_trackbars
        self.show_reconstruction = not options.no_show_reconstruction
        self.show_events = options.show_events
        self.event_display_mode = options.event_display_mode
        self.num_bins_to_show = options.num_bins_to_show
        self.gamma = options.gamma
        self.contrast = options.contrast
        self.brightness = options.brightness
        self.saturation = options.saturation

        self.window_name = 'E2VID'

        if self.display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

            if self.display_trackbars:
                # Gamma trackbar
                self.gamma_trackbar = Trackbar('Gamma', 0.5, 2.5, 40)
                cv2.createTrackbar(self.gamma_trackbar.name, self.window_name,
                                   self.gamma_trackbar.val_to_tick_pos(self.gamma), self.gamma_trackbar.num_ticks,
                                   self.on_gamma_changed)

                # Contrast trackbar
                self.contrast_trackbar = Trackbar('Contrast', 0.5, 2.0, 20)
                cv2.createTrackbar(self.contrast_trackbar.name, self.window_name,
                                   self.contrast_trackbar.val_to_tick_pos(
                                       self.contrast), self.contrast_trackbar.num_ticks,
                                   self.on_contrast_changed)

                # Brightness trackbar
                self.brightness_trackbar = Trackbar('Brightness', -50.0, 50.0, 100)
                cv2.createTrackbar(self.brightness_trackbar.name, self.window_name,
                                   self.brightness_trackbar.val_to_tick_pos(
                                       self.brightness), self.brightness_trackbar.num_ticks,
                                   self.on_brightness_changed)

                if options.color:
                    # Saturation trackbar
                    self.saturation_trackbar = Trackbar('Saturation', 0.0, 2.0, 30)
                    cv2.createTrackbar(self.saturation_trackbar.name, self.window_name,
                                       self.saturation_trackbar.val_to_tick_pos(
                                           self.saturation), self.saturation_trackbar.num_ticks,
                                       self.on_saturation_changed)

        self.border = options.display_border_crop
        self.wait_time = options.display_wait_time
        self.gamma_LUT = np.empty((1, 256), np.uint8)
        self.update_gamma_LUT(self.gamma)

    def update_gamma_LUT(self, gamma):
        for i in range(256):
            self.gamma_LUT[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    def on_gamma_changed(self, tick_pos):
        self.gamma = self.gamma_trackbar(tick_pos)
        print('Gamma: {:.2f}'.format(self.gamma))
        self.update_gamma_LUT(self.gamma)

    def on_contrast_changed(self, tick_pos):
        self.contrast = self.contrast_trackbar(tick_pos)
        print('Contrast: {:.2f}'.format(self.contrast))

    def on_brightness_changed(self, tick_pos):
        self.brightness = self.brightness_trackbar(tick_pos)
        print('Brightness: {:.2f}'.format(self.brightness))

    def on_saturation_changed(self, tick_pos):
        self.saturation = self.saturation_trackbar(tick_pos)
        print('Saturation: {:.2f}'.format(self.saturation))

    def crop_outer_border(self, img, border):
        if self.border == 0:
            return img
        else:
            return img[border:-border, border:-border]

    def __call__(self, img, events=None):

        if not self.display:
            return

        img = self.crop_outer_border(img, self.border)

        with Timer('Gamma correction'):
            if not self.gamma == 1.0:
                img = cv2.LUT(img, self.gamma_LUT)

        with Timer('Contrast/Brighntess correction'):
            if not (self.contrast == 1.0 and self.brightness == 0.0):
                cv2.convertScaleAbs(src=img, dst=img, alpha=self.contrast, beta=self.brightness)

        with Timer('Saturation correction'):
            img_is_color = (len(img.shape) == 3)
            if img_is_color and not self.saturation == 1.0:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
                (h, s, v) = cv2.split(img)
                s = s * self.saturation
                s = np.clip(s, 0, 255)
                img = cv2.merge([h, s, v])
                img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_HSV2BGR)

        if self.show_events:
            assert(events is not None)
            event_preview = make_event_preview(events, mode=self.event_display_mode,
                                               num_bins_to_show=self.num_bins_to_show)
            event_preview = self.crop_outer_border(event_preview, self.border)

        if self.show_events:
            img_is_color = (len(img.shape) == 3)
            preview_is_color = (len(event_preview.shape) == 3)

            if(preview_is_color and not img_is_color):
                img = np.dstack([img] * 3)
            elif(img_is_color and not preview_is_color):
                event_preview = np.dstack([event_preview] * 3)

            if self.show_reconstruction:
                img = np.hstack([event_preview, img])
            else:
                img = event_preview

        cv2.imshow(self.window_name, img)
        c = cv2.waitKey(self.wait_time)

        if c == ord('s'):
            now = datetime.now()
            path_to_screenshot = '/tmp/screenshot-{}.png'.format(now.strftime("%d-%m-%Y-%H-%M-%S"))
            cv2.imwrite(path_to_screenshot, img)
            print('Saving screenshot to: {}'.format(path_to_screenshot))
        elif c == ord('e'):
            self.show_events = not self.show_events
        elif c == ord('f'):
            self.show_reconstruction = not self.show_reconstruction

#############################################################################3

class DepthDisplay:
    """
    Utility class to display depth reconstructions
    """

    def __init__(self, options):
        self.display = options.display
        self.display_trackbars = not options.no_display_trackbars
        self.show_reconstruction = not options.no_show_reconstruction
        self.show_events = options.show_events
        self.event_display_mode = options.event_display_mode
        self.num_bins_to_show = options.num_bins_to_show

        self.window_name = 'EventsDepth'

        self.inv_depth = 0
        self.log_depth = 1

        if self.display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

            if self.display_trackbars:
                # create switch for ON/OFF functionality
                switch = 'INV'
                cv2.createTrackbar(switch, self.window_name,0,1,self.on_inv_depth)

                # create switch for ON/OFF functionality
                switch = 'LOG'
                cv2.createTrackbar(switch, self.window_name,1,1,self.on_log_depth)

        self.border = options.display_border_crop
        self.wait_time = options.display_wait_time
    
    def on_inv_depth(self, tick_pos):
        self.inv_depth = tick_pos

    def on_log_depth(self, tick_pos):
        self.log_depth = tick_pos

    def crop_outer_border(self, img, border):
        if self.border == 0:
            return img
        else:
            return img[border:-border, border:-border]

    def __call__(self, img, events=None):

        if not self.display:
            return

        img = self.crop_outer_border(img, self.border)
        
        with Timer('Inv Depth'):
            if self.inv_depth is 1:
                # Check if the image is in log depth
                if self.log_depth is 1:
                    # Convert to depth
                    img = np.exp(5.70378 * (img - np.ones((img.shape[0], img.shape[1]), dtype=np.float32)))
                    # Perform inverse depth
                    img = 1/img
                    img = img/np.amax(img)
                    #Convert back to log depth
                    img = np.ones((img.shape[0], img.shape[1]), dtype=np.float32) + np.log(img)/5.70378
                else:
                    # Perform inverse depth
                    img = 1/img
                    img = img/np.amax(img)

        with Timer('Log Depth'):
            if self.log_depth is 0:
                if self.inv_depth is 0:
                    # Perform norm depth (the prediction is already in log scale)
                    img = np.exp(5.70378 * (img - np.ones((img.shape[0], img.shape[1]), dtype=np.float32)))
                else: 
                    # Perform inverse depth
                    img = 1/img
                    img = img/np.amax(img)
                    # Perform norm depth (the prediction is already in log scale)
                    img = np.exp(5.70378 * (img - np.ones((img.shape[0], img.shape[1]), dtype=np.float32)))
                    img = 1/img
                    img = img/np.amax(img)
 

        if self.show_events:
            assert(events is not None)
            event_preview = make_event_preview(events, mode=self.event_display_mode,
                                               num_bins_to_show=self.num_bins_to_show)
            event_preview = self.crop_outer_border(event_preview, self.border)

        if self.show_events:
            img_is_color = (len(img.shape) == 3)
            preview_is_color = (len(event_preview.shape) == 3)

            if(preview_is_color and not img_is_color):
                img = np.dstack([img] * 3)
            elif(img_is_color and not preview_is_color):
                event_preview = np.dstack([event_preview] * 3)

            if self.show_reconstruction:
                img = np.hstack([event_preview, img])
            else:
                img = event_preview

        img = np.clip(img, 0.0, 1.0)
        cv2.imshow(self.window_name, img)
        c = cv2.waitKey(self.wait_time)

        if c == ord('s'):
            now = datetime.now()
            path_to_screenshot = '/tmp/screenshot-{}.png'.format(now.strftime("%d-%m-%Y-%H-%M-%S"))
            cv2.imwrite(path_to_screenshot, img)
            print('Saving screenshot to: {}'.format(path_to_screenshot))
        elif c == ord('e'):
            self.show_events = not self.show_events
        elif c == ord('f'):
            self.show_reconstruction = not self.show_reconstruction
