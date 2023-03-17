import os
import numpy as np
import cv2
import shutil
import random
import sys



IMG_SIZE = 128
SQUARE_LIMIT = 200
EXTEND_UPSCALE_LIMIT = 150
EXTEND_DOWNSCALE_LIMIT = 130
EXTEND_DOWNSCALE_INTERVAL = 5
EXTEND_UPSCALE_INTERVAL = 10
SQUARE_INTERVAL = 10
SCALE_FACTOR = 2

def convert_coords(box, img_shape):

	x, y, w, h = box[0], box[1], box[2], box[3]
	x1 = x - w/2.
	x2 = x + w/2.
	y1 = y - h/2.
	y2 = y + h/2.

	height, width, _ = img_shape
	x1, x2 = x1 * width, x2 * width
	y1, y2 = y1 * height, y2 * height

	if x1 < 0: x1 = 0
	if y1 < 0: y1 = 0
	if x2 > width: x2 = width -1
	if y2 > height: y2 = height - 1

	return [x1, y1, x2, y2]

def read_file(txt_path, img_path):
	cthresh = 0.0499
	f = open(txt_path, "r")
	boxes = f.readlines()
	f.close()

	img = cv2.imread(img_path)
	img_shape = img.shape
	new_boxes = []
	for box in boxes:
		box = box.strip()
		box = box.split(" ")
		class_id, x, y, w, h = int(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
		conf = None
		fp = None
		cbox = convert_coords([x, y, w, h], img_shape)

		if len(box) > 5:
			conf = float(box[5])
			# fp = int(box[6])
		#if conf is not None and conf < 0.5:
		#	continue
		#if class_id != 0:
		#	print("avengers")
		#	continue
		new_boxes.append([class_id, cbox, conf, fp])
		#print(new_boxes)
	return new_boxes

def get_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def get_boxes(imageName):

	txtName = imageName.replace(".jpg", ".txt")
	dataDir = "../data"
	imagesDir = os.path.join(dataDir, "images")
	labelsDir = os.path.join(dataDir, "labels")
	imagePath = os.path.join(imagesDir, imageName) 
	labelPath = os.path.join(labelsDir, txtName) 

	boxes = read_file(labelPath, imagePath)

	return boxes

def get_match(pred_box, gt_boxes, debug=False):

	pclass = 0
	pinfo = pred_box
	mask = np.zeros(len(gt_boxes))
	iou_thresh = 0.001

	for index, ginfo in enumerate(gt_boxes):
		gclass = ginfo[0]
		if pclass != gclass:
			continue
		iou = get_iou(ginfo[1], pinfo)
		if debug:
			print("IOU val: ", iou)
		if iou > iou_thresh:
			mask[index] = 1

	if np.sum(mask) > 1:
		return 0

	return 1

def get_index(pred_box, gt_boxes):

	pclass = 0
	pinfo = pred_box
	iou_thresh = 0.8
	max_iou, max_index = -1, -1

	for index, ginfo in enumerate(gt_boxes):
		gclass = ginfo[0]
		if pclass != gclass:
			continue
		iou = get_iou(ginfo[1], pinfo)
		if iou > max_iou:
			max_iou  = iou
			max_index = index

	if max_iou < iou_thresh: return -1

	return max_index

def get_box_from_image_name(image_name, images_dir):

	image_name = image_name.split("--")
	og_name = image_name[-1]

	og_path = os.path.join(images_dir, og_name)
	og_img = cv2.imread(og_path)
	h, w, c = og_img.shape

	image_shape = (h, w)
	image_name = image_name[0]
	image_name = image_name.split("_")

	cx = image_name[1]
	cy = image_name[2]
	cw = image_name[3]
	ch = image_name[4]

	cx, cy = float(cx), float(cy)
	cw, ch = float(cw), float(ch)

	norm_box = [cx, cy, cw, ch]

	pbox = get_crop_abs_coords(norm_box, image_shape)

	return pbox, image_shape


def get_random_check(thresh=50):

	check = 1
	rand_number = random.randint(0,100)
	if rand_number > thresh:
		check = 0

	return check

def make_image(img, new_shape, padding):

	new_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

	h, w = new_shape

	img = cv2.resize(img, (w, h))

	y, x = padding
	new_img[y:y+h, x:x+w] = img[:,:]

	return new_img 

def post_process(image, extended_box, new_shape, padding, person_box):

	tlx, tly = extended_box[0], extended_box[1]
	brx, bry = extended_box[2], extended_box[3]

	og_crop  = image[tly:bry, tlx:brx]
	big_crop = make_image(og_crop, new_shape, padding)
	person_mask  = make_mask(person_box)


	rect_top = (person_box[0], person_box[1])
	rect_bottom = (person_box[2], person_box[3])

	# Drawing rectangle over the image
	cv2.rectangle(big_crop, rect_top, rect_bottom, color = (0, 255, 0), thickness = 1)

	return big_crop, og_crop, person_mask

def make_mask(person_box):

	img = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)

	tlx = person_box[0]
	tly = person_box[1]
	brx = person_box[2]
	bry = person_box[3]

	img[tly:bry, tlx:brx]  = 255

	return img


# You can remove all the above functions

def get_match_with_index(pred_box, gt_boxes, box_index):

	pclass = gt_boxes[box_index][0]
	pinfo = pred_box
	mask = np.zeros(len(gt_boxes))
	iou_thresh = 0.001

	for index, ginfo in enumerate(gt_boxes):
		if index == box_index:
			continue

		gclass = ginfo[0]
		if pclass != gclass:
			continue

		iou = get_iou(ginfo[1], pinfo)

		if iou > iou_thresh:
			mask[index] = 1

	if np.sum(mask) > 0:
		return 0

	return 1


def get_extended_coords(img_shape, box, pcnt):

	px, py = pcnt
	h, w = img_shape
	cx, cy, cw, ch = box[0], box[1], box[2], box[3] 

	tlx = cx - px * (cw/2.)
	tly = cy - py * (ch/2.)
	brx = cx + px * (cw/2.)
	bry = cy + py * (ch/2.)

	tlx = int(tlx * w )
	tly = int(tly * h )
	brx = int(brx * w )
	bry = int(bry * h )

	if tlx <= 0:
		brx = brx - tlx
		tlx = 0
	if tly <= 0:
		bry = bry - tly
		tly = 0
	if brx >= w: 
		tlx = tlx - (brx - w + 1)
		brx = w - 1
	if bry >= h:
		tly = tly - (bry - h + 1)
		bry = h - 1

	if tlx <= 0: tlx = 0
	if tly <= 0: tly = 0
	if brx >= w: brx = w  - 1
	if bry >= h: bry = h - 1

	return [tlx, tly, brx, bry]

def make_it_square(crop_shape, pred_boxes, img_shape, box, box_index):

	crop_h, crop_w = crop_shape

	if crop_w > crop_h:
		req_ratio = int((float(crop_w)/float(crop_h)) * 100)
	else:
		req_ratio = int((float(crop_h)/float(crop_w)) * 100)

	req_ratio -= 100
	if req_ratio > SQUARE_LIMIT: req_ratio = SQUARE_LIMIT

	start, end, interval = 0, req_ratio, SQUARE_INTERVAL

	temp = None

	for ratio in range(start, end, interval):
		extVal = float(ratio)/100.0

		if crop_w > crop_h:
			px = 1 
			py = 1 + extVal
		else: 
			py = 1 
			px = 1 + extVal

		pcnt = (px, py)
		cbox = get_extended_coords(img_shape, box, pcnt)
		check = get_match_with_index(cbox, pred_boxes, box_index)

		if check == 0 and temp is None:
			return cbox, 0
		elif check == 0 and temp is not None:
			return temp, 1
		elif check == 1:
			temp = cbox

	return temp, 2



def normalize_coords(box, img_shape):

	tlx, tly, brx, bry = box[0], box[1], box[2], box[3]
	h, w = img_shape

	crop_w, crop_h = brx - tlx, bry - tly

	cx = tlx + crop_w/2.
	cy = tly + crop_h/2.

	cx = float(cx)/float(w) 
	cy = float(cy)/float(h)
	cw = float(crop_w)/float(w)
	ch = float(crop_h)/float(h)

	crop_shape = (crop_h, crop_w)

	return [cx, cy, cw, ch], crop_shape

def get_crop_abs_coords(box, img_shape):

	cx, cy, cw, ch = box[0], box[1], box[2], box[3]
	h, w = img_shape

	x, y = cx - cw/2., cy - ch/2.

	x1 = int(x * w)
	y1 = int(y * h)
	x2 = int(cw * w ) + x1
	y2 = int(ch * h ) + y1

	if x1 < 0: x1 = 0
	if y1 < 0: y1 = 0
	if x2 >= w: x2 = w - 1
	if y2 >= h: y2 = h - 1

	return [x1, y1, x2, y2]

def extend_crop(crop_shape, pred_boxes, img_shape, box, box_index):

	crop_w, crop_h = crop_shape

	max_wh = 0
	if crop_w > crop_h:
		max_wh = crop_w
	else:
		max_wh = crop_h

	req_ratio = int((float(IMG_SIZE)/float(max_wh)) * 100)

	start, end, interval = -1, -1, 1

	if req_ratio <= 100:
		start, end, interval = 100, EXTEND_DOWNSCALE_LIMIT, EXTEND_DOWNSCALE_INTERVAL
	else:
		if req_ratio > EXTEND_UPSCALE_LIMIT:
			req_ratio = EXTEND_UPSCALE_LIMIT
		start, end, interval = 100, req_ratio, EXTEND_UPSCALE_INTERVAL


	temp = None

	for ratio in range(start, end, interval):

		extVal = float(ratio)/100.0
		px = extVal
		py = extVal
		pcnt = (px, py)

		cbox = get_extended_coords(img_shape, box, pcnt)
		check = get_match_with_index(cbox, pred_boxes, box_index)

		if check == 0 and temp is None:
			return cbox, 0
		elif check == 0 and temp is not None:
			return temp, 1
		elif check == 1: 
			temp = cbox

	return temp, 2

def get_modified_box(norm_pbox, new_shape, padding):

	abs_pbox = get_crop_abs_coords(norm_pbox, new_shape)

	pad_y, pad_x = padding

	tlx, tly = abs_pbox[0], abs_pbox[1]
	brx, bry = abs_pbox[2], abs_pbox[3]

	tlx += pad_x
	tly += pad_y
	brx += pad_x
	bry += pad_y

	if tlx < 0: tlx = 0
	if tly < 0: tly = 0
	if brx > IMG_SIZE: brx = IMG_SIZE - 1
	if bry > IMG_SIZE: bry = IMG_SIZE - 1

	return [tlx, tly, brx, bry] 

def resize_and_pad(img_crop):

	crop_tlx, crop_tly = img_crop[0], img_crop[1]
	crop_brx, crop_bry = img_crop[2], img_crop[3]

	crop_w, crop_h = crop_brx - crop_tlx, crop_bry - crop_tly

	scale = 1

	if crop_w >  crop_h:
		scale = IMG_SIZE/float(crop_w)
	else:
		scale = IMG_SIZE/float(crop_h)

	if scale > SCALE_FACTOR: scale =   SCALE_FACTOR

	new_h = crop_h * scale
	new_w = crop_w * scale
	new_h, new_w = int(new_h), int(new_w)

	if new_h > IMG_SIZE: new_h = IMG_SIZE
	if new_w > IMG_SIZE: new_w = IMG_SIZE

	pad_x = IMG_SIZE - new_w
	pad_y = IMG_SIZE - new_h

	pad_left = int(pad_x / 2)
	pad_top =  int(pad_y / 2)
	# pad_left = random.randint(0, pad_x) 
	# pad_top  = random.randint(0, pad_y) 

	return scale, (new_h, new_w), (pad_top, pad_left)

def get_box_info(image_shape, pred_boxes, box_index):


	#pbox = person_box
	pbox = pred_boxes[box_index][1]
	person_w, person_h = pbox[2] - pbox[0], pbox[3] - pbox[1]
	person_shape = (person_h, person_w)

	norm_box, _ = normalize_coords(pbox, image_shape)
	square_box, state = make_it_square(person_shape, pred_boxes, image_shape, norm_box, box_index)

	if square_box is None:
		return None, None, None, None, None

	norm_square_box, square_box_shape = normalize_coords(square_box, image_shape)
	extended_box, state = extend_crop(square_box_shape, pred_boxes, image_shape, norm_square_box, box_index)

	if extended_box is None:
		return None, None, None, None, None

	tlx, tly, brx, bry = extended_box[0], extended_box[1], extended_box[2], extended_box[3]
	x1, y1, x2, y2 = pbox[0], pbox[1], pbox[2], pbox[3]
	pbox = [x1 - tlx, y1 - tly, x2 - tlx, y2 - tly]
	crop_shape = (bry - tly, brx - tlx)
	
	norm_pbox , _ = normalize_coords(pbox, crop_shape)
	scale, new_shape, padding = resize_and_pad(extended_box)
	new_pbox = get_modified_box(norm_pbox, new_shape, padding)

	return extended_box, new_shape, padding, new_pbox, scale


'''
data_dir = sys.argv[1]
images_dir = "../data/images"
save_dir = "context"
og_dir = "original"
masks_dir =  sys.argv[2]
images = os.listdir(data_dir)

counter = 0
t2 = 0
for image in images:

	og_crop_name = image
	og_name = image.split("--")[-1]
	img_path = os.path.join(data_dir, og_crop_name)
	img = cv2.imread(img_path)
	h, w, c = img.shape
	full_image_path = os.path.join(images_dir, og_name)
	full_image = cv2.imread(full_image_path)
	#if h * w <= 600:
	#	continue

	person_box, image_shape = get_box_from_image_name(image, images_dir)
	pred_boxes = get_boxes(og_name)
	box_index = get_index(person_box, pred_boxes)

	if box_index == -1:
		continue

	#[[class_id, [tlx, tly, brx, bry]]] - format of pred_boxes
	# box_index - 0 based index for the box that we want to get the extended crop
	# image_shape - original image shape in the format (h, w)
	extended_box, new_shape, padding, person_box, scale = get_box_info(image_shape, pred_boxes, box_index)
	# extended_box - [x2, y2, x3, y3]
	# person_box - [X0', Y0', X1', Y1']
	# scale - Scale factor for resizing
	# new_shape - (h, w) of the resized image
	# padding - (pad_left, pad_top)

	if extended_box is None:
		continue

	img_crop, og_crop, mask = post_process(full_image, extended_box, new_shape, padding, person_box)

	if img_crop is None:
		continue

	post_h, post_w, post_c = img_crop.shape
	if post_h * post_w < 600:
		continue

	write_path = os.path.join(save_dir, image)
	og_path = os.path.join(og_dir, image)
	mask_path = os.path.join(masks_dir, image)
	cv2.imwrite(write_path, img_crop)
	cv2.imwrite(og_path, og_crop)
	cv2.imwrite(mask_path, mask)
	counter += 1
	#shutil.copy(img_path, og_dir)
	if counter > 100:
	 	break

'''