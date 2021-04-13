front_matter = """
------------------------------------------------------------------------
Online demo for [LoFTR](https://zju3dv.github.io/loftr/).

This demo is heavily inspired by [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork/).
We thank the authors for their execellent work.
------------------------------------------------------------------------
"""

import os
import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.cm as cm

os.sys.path.append("../")  # Add the project directory
from src.loftr import LoFTR, default_cfg
from src.config.default import get_cfg_defaults
try:
    from demo.utils import (AverageTimer, VideoStreamer,
                            make_matching_plot_fast, make_matching_plot, frame2tensor)
except:
    raise ImportError("This demo requires utils.py from SuperGlue, please use run_demo.sh to start this script.")


torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LoFTR online demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weight', type=str, help="Path to the checkpoint.")
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')
    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--save_video', action='store_true',
        help='Save output (with match visualizations) to a video.')
    parser.add_argument(
        '--save_input', action='store_true',
        help='Save the input images to a video (for gathering repeatable input source).')
    parser.add_argument(
        '--skip_frames', type=int, default=1, 
        help="Skip frames from webcam input.")
    parser.add_argument(
        '--top_k', type=int, default=2000, help="The max vis_range (please refer to the code).")
    parser.add_argument(
        '--bottom_k', type=int, default=0, help="The min vis_range (please refer to the code).")

    opt = parser.parse_args()
    print(front_matter)
    parser.print_help()

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        raise RuntimeError("GPU is required to run this demo.")

    # Initialize LoFTR
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(opt.weight)['state_dict'])
    matcher = matcher.eval().to(device=device)

    # Configure I/O
    if opt.save_video:
        print('Writing video to loftr-matches.mp4...')
        writer = cv2.VideoWriter('loftr-matches.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (640*2 + 10, 480))
    if opt.save_input:
        print('Writing video to demo-input.mp4...')
        input_writer = cv2.VideoWriter('demo-input.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))

    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    frame_id = 0  
    last_image_id = 0
    frame_tensor = frame2tensor(frame, device)
    last_data = {'image0': frame_tensor}
    last_frame = frame

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        window_name = 'LoFTR Matches'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, (640*2, 480))
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the reference image (left)\n'
          '\td/f: move the range of the matches (ranked by confidence) to visualize\n'
          '\tc/v: increase/decrease the length of the visualization range (i.e., total number of matches) to show\n'
          '\tq: quit')

    timer = AverageTimer()
    vis_range = [opt.bottom_k, opt.top_k]

    while True:
        frame_id += 1
        frame, ret = vs.next_frame()
        if frame_id % opt.skip_frames != 0:
            # print("Skipping frame.")
            continue
        if opt.save_input:
            inp = np.stack([frame]*3, -1)
            inp_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            input_writer.write(inp_rgb)
        if not ret:
            print('Finished demo_loftr.py')
            break
        timer.update('data')
        stem0, stem1 = last_image_id, vs.i - 1

        frame_tensor = frame2tensor(frame, device)
        last_data = {**last_data, 'image1': frame_tensor}
        matcher(last_data)

        total_n_matches = len(last_data['mkpts0_f'])
        mkpts0 = last_data['mkpts0_f'].cpu().numpy()[vis_range[0]:vis_range[1]]
        mkpts1 = last_data['mkpts1_f'].cpu().numpy()[vis_range[0]:vis_range[1]]
        mconf = last_data['mconf'].cpu().numpy()[vis_range[0]:vis_range[1]]

        # Normalize confidence.
        if len(mconf) > 0:
            conf_vis_min = 0.
            conf_min = mconf.min()
            conf_max = mconf.max()
            mconf = (mconf - conf_vis_min) / (conf_max - conf_vis_min + 1e-5)

        timer.update('forward')
        alpha = 0
        color = cm.jet(mconf, alpha=alpha)

        text = [
            f'LoFTR',
            '# Matches (showing/total): {}/{}'.format(len(mkpts0), total_n_matches),
        ]
        small_text = [
            f'Showing matches from {vis_range[0]}:{vis_range[1]}',
            f'Confidence Range: {conf_min:.2f}:{conf_max:.2f}',
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]
        out = make_matching_plot_fast(
            last_frame, frame, mkpts0, mkpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=False, small_text=small_text)

        # Save high quality png, optionally with dynamic alpha support (unreleased yet).
        # save_path = 'demo_vid/{:06}'.format(frame_id)
        # make_matching_plot(
        #     last_frame, frame, mkpts0, mkpts1, mkpts0, mkpts1, color, text,
        #     path=save_path, show_keypoints=opt.show_keypoints, small_text=small_text)

        if not opt.no_display:
            if opt.save_video:
                writer.write(out)
            cv2.imshow('LoFTR Matches', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                if opt.save_video:
                    writer.release()
                if opt.save_input:
                    input_writer.release()
                vs.cleanup()
                print('Exiting...')
                break
            elif key == 'n':  
                last_data['image0'] = frame_tensor
                last_frame = frame
                last_image_id = (vs.i - 1)
                frame_id_left = frame_id
            elif key in ['d', 'f']:
                if key == 'd':
                    if vis_range[0] >= 0:
                       vis_range[0] -= 200
                       vis_range[1] -= 200
                if key =='f':
                    vis_range[0] += 200
                    vis_range[1] += 200
                print(f'\nChanged the vis_range to {vis_range[0]}:{vis_range[1]}')
            elif key in ['c', 'v']:
                if key == 'c':
                    vis_range[1] -= 50
                if key =='v':
                    vis_range[1] += 50
                print(f'\nChanged the vis_range[1] to {vis_range[1]}')
        elif opt.output_dir is not None:
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)
        else:
            raise ValueError("output_dir is required when no display is given.")
        timer.update('viz')
        timer.print()


    cv2.destroyAllWindows()
    vs.cleanup()
