class Segments(object):
    def find_continous_segments1(self,array):
        """Find the continous segments here"""
        if len(array) == 0:
            return []
        prev = array[0]
        segments = []
        start = array[0]
        end = array[0]
        for now in array[1:]:
            if now == prev + 1:
                end = now
            else:
                each_segment = [start,end]
                segments.append(each_segment)
                start = now
                end = now
            prev = now
        each_segment = [start, end]
        segments.append(each_segment)
        return segments

    def find_continous_segments(self, array,l):
        """Find the continous segments here"""
        if len(array) == 0:
            return []
        prev = array[0]
        segments = []
        start = array[0]
        end = array[0]
        for now in array[1:]:
            if now == prev + 1:
                end = now
            else:
                start = l[start][0]
                end = l[end][1]
                each_segment = [start, end]
                segments.append(each_segment)
                start = now
                end = now
            prev = now
        each_segment = [l[start][0], l[end][1]]
        segments.append(each_segment)
        return segments

# c = Segments()
# print(c.find_continous_segments([0, 1, 2, 4,5],[[1,2],[2,3],[3,4],[4,5],[6,7],[7,8]]))