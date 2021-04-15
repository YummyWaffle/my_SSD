import torch
import math

class PriorBox(object):
    featmap_size = {
        300: (38,19,10,5,3,1),
        512: (64,32,16,8,4,1)
    }
    def __init__(self,s_min=0.2,s_max=0.9,aspect_ratio=[[2],[2,3],[2,3],[2,3],[2],[2]],ssd_version=300):
        assert(len(aspect_ratio)==6)
        assert(s_min > 0)
        assert(s_max < 1)
        self.ssd_version = ssd_version
        self.aspect_ratio = aspect_ratio
        # calculate default size
        self.default_size  = []
        for i in range(1,7):
            s_k = s_min+(s_max-s_min)*(i-1)/(6-1)
            version_specific_size = s_k * self.ssd_version
            self.default_size.append(version_specific_size)
        self.default_size.append(self.ssd_version)
    
    def __call__(self):
        all_box = []
        featmap_size_list = self.featmap_size[self.ssd_version]
        for feat_map_index in range(6):
            scale_feat_size = featmap_size_list[feat_map_index]
            for i in range(scale_feat_size):
                for j in range(scale_feat_size):
                    #c_x = (i + 0.5) * (self.ssd_version/scale_feat_size) / self.ssd_version
                    c_x = (i + 0.5) / scale_feat_size
                    #c_y = (j + 0.5) * (self.ssd_version/scale_feat_size) / self.ssd_version
                    c_y = (j + 0.5) / scale_feat_size
                    # Generate Default Box : s_k & sqrt(s_k * s_k+1)
                    default_1 = self.default_size[feat_map_index]
                    default_2 = math.sqrt(self.default_size[feat_map_index] * self.default_size[feat_map_index+1])
                    all_box.append([c_x,c_y,default_1/self.ssd_version,default_1/self.ssd_version])
                    all_box.append([c_x,c_y,default_2/self.ssd_version,default_2/self.ssd_version])
                    # Generate Aspect Ratio Box
                    for ratios in self.aspect_ratio[feat_map_index]:
                        width_trans1 = default_1 * math.sqrt(ratios)
                        height_trans1 = default_1 / math.sqrt(ratios)
                        width_trans2 = default_1 * math.sqrt(1/ratios)
                        height_trans2 = default_1 / math.sqrt(1/ratios)
                        all_box.append([c_x,c_y,width_trans1/self.ssd_version,height_trans1/self.ssd_version])
                        all_box.append([c_x,c_y,width_trans2/self.ssd_version,height_trans2/self.ssd_version])
        output = torch.Tensor(all_box).view(-1,4)
        output.clamp_(max=1,min=0)
        return output

if __name__ == '__main__':
    box = PriorBox()
    all_box = box()
    print(all_box.size())