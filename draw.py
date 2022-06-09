import matplotlib.pyplot as plt
def draw(*, guting, banquio, zhongli, xitun, tainan, cianjhen):
	img = plt.imread("taiwan.png")
	fig, ax = plt.subplots()
	ax.imshow(img)
	x = [160, 150, 140, 110, 95, 100]
	y = [40, 43, 45, 90, 140, 170]
	notation = [guting, banquio, zhongli, xitun, tainan, cianjhen]
	# ax.imshow(img, extent=[0, 400, 0, 300])
	ax.scatter(x, y)
	for i, txt in enumerate(notation):
	    ax.annotate(txt, (x[i], y[i]))
	plt.show()

if __name__ == "__main__":
	draw(guting = 0, banquio = 0, zhongli = 0, xitun = 0, tainan = 0, cianjhen = 0)