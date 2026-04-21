import cv2
import numpy as np
import time

WIN_MAIN = "Invisibility"
WIN_MASK = "Mask (debug)"
WIN_CTRL = "Controls"

def nothing(x):
    pass

def ensure_odd(x):
    return x if x % 2 == 1 else max(1, x-1)

def average_background(cap, frames=40):
    # Capture and average frames (used when user presses 'B')
    acc = None
    count = 0
    for i in range(frames):
        ret, f = cap.read()
        if not ret:
            continue
        f = cv2.flip(f, 1)
        if acc is None:
            acc = np.float32(f)
        else:
            cv2.accumulate(f, acc)
        count += 1
    if acc is None or count == 0:
        return None
    avg = (acc / count).astype(np.uint8)
    return avg

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    cv2.namedWindow(WIN_MAIN)
    cv2.namedWindow(WIN_MASK)
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 500, 300)

    # Trackbars for HSV tuning and morphological settings
    cv2.createTrackbar('H low', WIN_CTRL, 0, 179, nothing)
    cv2.createTrackbar('H high', WIN_CTRL, 10, 179, nothing)
    cv2.createTrackbar('S low', WIN_CTRL, 120, 255, nothing)
    cv2.createTrackbar('V low', WIN_CTRL, 70, 255, nothing)
    cv2.createTrackbar('Kernel', WIN_CTRL, 3, 21, nothing)        # morphological kernel
    cv2.createTrackbar('Blur', WIN_CTRL, 21, 101, nothing)        # gaussian blur kernel (odd)
    cv2.createTrackbar('MaskThresh', WIN_CTRL, 127, 255, nothing) # final mask threshold
    cv2.createTrackbar('DiffThresh', WIN_CTRL, 30, 255, nothing) # threshold for "Full" mode

    print("Instructions:")
    print(" - Press 'b' to capture a single background frame (keep cloak OUT of frame).")
    print(" - Press 'B' to capture an AVG background (better, keep cloak OUT).")
    print(" - Tune H low / H high / S low / V low until the cloak appears white in the Mask window.")
    print(" - Press 'i' to toggle replacing with an image instead of captured background.")
    print(" - Press 'm' to toggle between Color Cloak and FULL INVISIBILITY mode.")
    print(" - Press 'q' to quit.\n")

    background = None
    bg_image = None
    use_image_bg = False
    use_diff_mode = False # Toggle for Full Invisibility

    # OPTIONAL: Load a replacement image (resize to camera size when loaded)
    replacement_path = None  # set to a file path if you want a default image


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1) 

        h_low = cv2.getTrackbarPos('H low', WIN_CTRL)
        h_high = cv2.getTrackbarPos('H high', WIN_CTRL)
        s_low = cv2.getTrackbarPos('S low', WIN_CTRL)
        v_low = cv2.getTrackbarPos('V low', WIN_CTRL)
        ksize = cv2.getTrackbarPos('Kernel', WIN_CTRL)
        blur_k = ensure_odd(cv2.getTrackbarPos('Blur', WIN_CTRL))
        mask_thresh = cv2.getTrackbarPos('MaskThresh', WIN_CTRL)
        diff_thresh = cv2.getTrackbarPos('DiffThresh', WIN_CTRL)

        # constrain sizes
        ksize = max(1, ksize)
        if ksize % 2 == 0:
            ksize += 1

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if use_diff_mode and background is not None:
            # FULL INVISIBILITY Mode: Anything different from background
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            
            # handle background resize if needed
            if gray_bg.shape != gray_frame.shape:
                gray_bg = cv2.resize(gray_bg, (gray_frame.shape[1], gray_frame.shape[0]))
                
            diff = cv2.absdiff(gray_frame, gray_bg)
            _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
        else:
            # COLOR CLOAK Mode: HSV range detection
            lower1 = np.array([h_low, s_low, v_low], dtype=np.uint8)
            upper1 = np.array([h_high, 255, 255], dtype=np.uint8)

            if h_low <= h_high:
                mask = cv2.inRange(hsv, lower1, upper1)
            else:
                lowerA = np.array([h_low, s_low, v_low], dtype=np.uint8)
                upperA = np.array([179, 255, 255], dtype=np.uint8)
                lowerB = np.array([0, s_low, v_low], dtype=np.uint8)
                upperB = np.array([h_high, 255, 255], dtype=np.uint8)
                maskA = cv2.inRange(hsv, lowerA, upperA)
                maskB = cv2.inRange(hsv, lowerB, upperB)
                mask = cv2.bitwise_or(maskA, maskB)

        # morphological cleanup
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # smooth edges
        if blur_k > 1:
            mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

        # final threshold to make mask binary and sharp
        _, mask = cv2.threshold(mask, mask_thresh, 255, cv2.THRESH_BINARY)

        mask_inv = cv2.bitwise_not(mask)

        # background selection: captured background or replacement image
        if use_image_bg and bg_image is not None:
            bg = bg_image
        else:
            bg = background

        # if background exists, composite; else just show frame + mask preview
        if bg is not None:
            # ensure same size
            if bg.shape[:2] != frame.shape[:2]:
                bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))

            # extract areas
            cloak_part = cv2.bitwise_and(bg, bg, mask=mask)          # where cloak -> show background
            non_cloak = cv2.bitwise_and(frame, frame, mask=mask_inv) # everything else -> show frame
            output = cv2.add(cloak_part, non_cloak)
        else:
            output = frame.copy()

        # small mask preview (BGR)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        preview = cv2.resize(mask_bgr, (int(frame.shape[1]*0.25), int(frame.shape[0]*0.25)))
        # place preview at top-left of output
        h0, w0 = preview.shape[:2]
        output[0:h0, 0:w0] = preview

        # helpful on-screen text
        mode_name = "FULL INVISIBLE" if use_diff_mode else "COLOR CLOAK"
        txt = f"[{mode_name}] b/B: capture BG  |  m: toggle mode  |  q: quit"
        cv2.putText(output, txt, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

        cv2.imshow(WIN_MAIN, output)
        cv2.imshow(WIN_MASK, mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            # capture single frame background 
            ret_bg, bgf = cap.read()
            if ret_bg:
                background = cv2.flip(bgf, 1)
                print("[info] single background captured (keep cloak OUT of frame)")
        elif key == ord('B'):
            print("[info] capturing averaged background (keep cloak OUT of frame)...")
            background = average_background(cap, frames=40)
            if background is not None:
                print("[info] averaged background captured")
            else:
                print("[warn] failed to capture averaged background")
        elif key == ord('m'):
            use_diff_mode = not use_diff_mode
            print(f"[info] Mode changed to: {'FULL INVISIBLE' if use_diff_mode else 'COLOR CLOAK'}")
            if use_diff_mode and background is None:
                print("[warn] Background not captured! Please press 'B' first.")
        elif key == ord('i'):
            # toggle image background on/off; lazy-load if needed
            use_image_bg = not use_image_bg
            if use_image_bg and bg_image is None:
                # prompt user to type path in console 
                print("Enter path to replacement image file (or press Enter to skip): ")
                path = input().strip()
                if path:
                    img = cv2.imread(path)
                    if img is None:
                        print("Could not load image:", path)
                        use_image_bg = False
                    else:
                        bg_image = img
                        print("Replacement image loaded.")
                else:
                    print("No path entered — keeping captured background (if any).")
                    use_image_bg = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
