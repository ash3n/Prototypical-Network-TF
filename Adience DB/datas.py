import numpy as np
import pandas as pd
import utilys as ut
import random

def get_data(lower=4,upper=10):

	dfs = [ ut.load_txt_as_df('data/fold/fold_0_data.txt'),
		ut.load_txt_as_df('data/fold/fold_1_data.txt'),
		ut.load_txt_as_df('data/fold/fold_2_data.txt'),
		ut.load_txt_as_df('data/fold/fold_3_data.txt'),
		ut.load_txt_as_df('data/fold/fold_4_data.txt'), ]

	df = pd.concat(dfs)
	l_faceid = list(df['face_id'])

	unique = sorted([int(x)-1 for x in set(df['face_id'])])
	n_faces = len(unique)
	m_df = len(df)

	inds = [ [] for _ in range(n_faces) ]
	for i in range(m_df):
		inds[int(l_faceid[i])-1].append(i)

	counts = [len(x) for x in inds]
	slices = [x//upper+int(x<upper) for x in counts]

	pre_order = counts[:]
	ordered = []
	for _ in range(len(unique)):
		curr_max = max(pre_order)
		curr_top = pre_order.index(curr_max)
		if curr_max >= lower:
			ordered.append(curr_top)
		pre_order[curr_top] = -1

	inds = [x for x in inds if len(x)>=lower]
	slices = [v for i, v in enumerate(slices) if counts[i]>=lower]
	counts = [x for x in counts if x>=lower]

	return df, inds, counts, slices, ordered

def load_imgs(df,width=100):
	imgs = []
	for i in range(len(df)):
		if i%1000 == 0:
			print(i,'images loaded..')
		img_path = ut.format_from_index(df,i)
		imgs.append(ut.load_image(img_path,(width,width)))
	return imgs

def gen_epoch(inds,counts,slices,ordered,imgs,n_classes=5,lower=4,upper=10):

	n_slices = sum(slices)
	n_groups = n_slices//n_classes

	pre_sort = []
	for i, v in enumerate(slices):
		for _ in range(v):
			pre_sort.append(i)

	post_sort = []
	for i_group in range(n_groups):
		no_dupes = pre_sort[:]
		this_group = []
		for i_class in range(n_classes):
			if len(no_dupes) > n_classes-i_class:
				rand_ind = random.randint(0,len(no_dupes)-1)
				rand_val = no_dupes[rand_ind]
				no_dupes = [x for x in no_dupes if x!=rand_val]
				this_group.append(rand_val)
				pre_sort.remove(rand_val)
		post_sort.append(this_group)

	post_sort = [x for x in post_sort if len(x)==n_classes]

	pre_load_inds = [x for x in inds if len(x)>=lower]
	pre_load_slices = slices[:]

	X_list = []
	y_list = []
	for i_group, v_group in enumerate(post_sort):
		X_append, y_append = [], []
		for i_class, v_class in enumerate(v_group):
			start_at = (pre_load_slices[v_class]-1)*upper
			class_inds = pre_load_inds[v_class][start_at:]
			pre_load_inds[v_class] = pre_load_inds[v_class][:start_at-1]
			pre_load_slices[v_class] -= 1
			for x in class_inds:
				y_append.append(ut.one_hot(i_class,5))
				X_append.append(imgs[x])
		X_list.append(X_append)
		y_list.append(y_append)

	return X_list, y_list