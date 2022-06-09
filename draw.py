import matplotlib.pyplot as plt
def draw(guting, banquio, zhongli, xitun, tainan, cianjhen):
	img = plt.imread("taiwan.png")
	fig, ax = plt.subplots()
	ax.imshow(img)
	x = [160, 150, 140, 110, 95, 100]
	y = [40, 43, 45, 90, 140, 170]
	notation = [guting, banquio, zhongli, xitun, tainan, cianjhen]
	# ax.imshow(img, extent=[0, 400, 0, 300])
	ax.scatter(x, y)
	for i, txt in enumerate(notation):
		if i == 1:
			ax.annotate(txt, (x[i]+1, y[i]+2))
		elif i == 2:
			ax.annotate(txt, (x[i], y[i] + 5))
		else:
			ax.annotate(txt, (x[i], y[i]))
	plt.show()

if __name__ == "__main__":
	draw(0.2916792631149292, 0.40774011611938477, 0.7007879018783569, 0.27297237515449524, 0.2524319291114807, 0.4588162302970886)