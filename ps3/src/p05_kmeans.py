from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import time

def k_means(points, K):
    """
    Arguments:
        - points: (M, N)
    """
    
    #initialization
    M, N = points.shape
    centroid = points[np.random.choice(range(M),K,False),:]
        
    #record ith point belongs to which cluster
    belong = np.zeros(M)
    max_iter = 300
    it = 0
    
    #change uint8 to float64 to enable norm calculation
    points = points.astype(np.float64)
    centroid = centroid.astype(np.float64)
    
    while it < max_iter:
        loss = 0
        pre_belong = belong.copy()
        # find the nearest centriod to belong
        dist = np.linalg.norm(points[:,None] - centroid[None], axis = 2)
        belong = np.argmin(dist, axis = 1)
        
        for j in range(K):
            # update centriod
            centroid[j] = np.mean(points[belong == j], axis = 0)
            
            #calculate loss
            loss += np.linalg.norm(points[belong == j] - centroid[j]) ** 2
        if (pre_belong == belong).all():
            break
        else:
            it = it + 1
            print('iteration: {}, loss: {}'.format(it,loss))
            continue
    
    centroid = centroid.astype(np.uint8)
    for j in range(K):
        points[belong == j] = centroid[j]
    points = points.astype(np.uint8)
    return centroid, points
            

if __name__ == '__main__':
    time1 = time.time()
    small_path = '../data/peppers-small.tiff'
    large_path = '../data/peppers-large.tiff'

    small = imread(small_path)
    large = imread(large_path)
    K = 16
    
    # L, W, H = small.shape
    # v_small = small.reshape((-1,H))
    # _, flattened_new_small = k_means(v_small, 16)
    # new_small = flattened_new_small.reshape((L, W, H))
    # plt.imshow(new_small)
    
    L, W, H = large.shape
    v_large = large.reshape((-1,H))
    _, flattened_new_large = k_means(v_large, 16)
    new_large = flattened_new_large.reshape((L, W, H))
    
    plt.subplot(1, 2, 1)
    plt.imshow(large)
    plt.subplot(1, 2, 2)
    plt.imshow(new_large)
    plt.savefig("compressed.png")
    time2 = time.time()
    print(time2 - time1)

# for i, image in enumerate([large,new_large]):
#     plt.imshow(image)
#     plt.axis('off')
#     fig = plt.gcf()
#     fig.set_size_inches(7.0/3,7.0/3) #dpi = 300, output = 700*700 pixels
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
#     plt.margins(0,0)
#     fig.savefig('{}.png'.format(i), format='png', transparent=True, dpi=300, pad_inches = 0)