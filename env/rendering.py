import cv2
import numpy as np

from env.utils import haversine_coord


text_kwargs = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=1.0,
                   color=(255, 255, 255),
                   thickness=2)
colors = [
    (255, 0, 255),  # 紫色
    (255, 100, 0),  # 蓝色
    (0, 165, 255),  # 橙色
    (255, 255, 0),  # 青色/黄色偏青

    (0, 255, 255),  # 黄色/青色偏黄
]


def make_random_color():
    r = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    b = np.random.randint(0, 256)
    return b, g, r


def value_to_rgb(value, min_val=1 / 1.5, max_val=1 / 0.1):
    # 限制范围
    if value >= max_val: return 0, 0, 255  # BGR: 红色
    if value <= min_val: return 0, 255, 0  # BGR: 绿色

    # 归一化到 [0, 1]
    t = (value - min_val) / (max_val - min_val)
    if t <= 0.5:  # 绿 → 黄 → 红
        # 绿色到黄色：增加红色通道，绿色通道保持 255
        r = int(255 * (t / 0.5))  # 0 → 255
        g, b = 255, 0
    else:
        # 黄色到红色：减少绿色通道，红色通道保持 255
        g = int(255 * (1 - (t - 0.5) / 0.5))  # 255 → 0
        r, b = 255, 0
    return b, g, r  # OpenCV BGR 顺序

# def value_to_rgb(value, min_val=1/1.5, max_val=1/0.1):
#     if value >= max_val: return 0, 0, 255  # 限制最大值
#
#     t = (value - min_val) / (max_val - min_val)  # 归一化 [0, 1]
#     r, g, b = 255, int(255 * (1 - t / 0.5)), 0
#     return b, g, r


class StaticRender:
    def __init__(self, graph, height=1440, padding=0.075):
        self.graph = graph

        lats = [graph.nodes[node]['y'] for node in graph.nodes]
        longs = [graph.nodes[node]['x'] for node in graph.nodes]
        self.bounder = (min(lats), max(lats), min(longs), max(longs))

        delta_lat = (self.bounder[3] - self.bounder[2])
        delta_lon = (self.bounder[1] - self.bounder[0])
        width = int(height * delta_lat / delta_lon)
        self.padding = (int(width * padding), int(height * padding))
        self.size = (width, height)
        self.density = height / haversine_coord(p1=(self.bounder[0], self.bounder[2]),
                                                p2=(self.bounder[1], self.bounder[2]),
                                                km=False)
        self.base_img = None

    def latlon_to_pixel(self, u):
        node_info = self.graph.nodes[u]
        lat, lon = node_info['y'], node_info['x']
        min_lat, max_lat, min_lon, max_lon = self.bounder
        pad_w, pad_h = self.padding
        w_ = self.size[0] - pad_w * 2
        h_ = self.size[1] - pad_h * 2

        x = int((lon - min_lon) / (max_lon - min_lon) * w_)
        y = h_ - int((lat - min_lat) / (max_lat - min_lat) * h_)
        return x+pad_w, y+pad_h

    def draw_nodes(self, nodes=None, color=(0, 255, 0), radius=2):
        if nodes is None: nodes = {}

        for i, u in enumerate(self.graph.nodes):
            pt = self.latlon_to_pixel(u)
            color_ = (0, 255, 255) if u in nodes.keys() else color
            # cv2.circle(self.base_img, pt, radius, color_, -1)
            # if i % 50 == 0:
            #     cv2.putText(self.base_img, f'{i}', org=pt, **text_kwargs)

        text = 'Nodes: {}'.format(len(self.graph.nodes))
        if len(nodes) > 0:
            text += '({})'.format(len(nodes))
        cv2.putText(self.base_img, text, org=(60, 60), **text_kwargs)

    def draw_edges(self, edges=None, color=(0, 255, 0), thickness=1):
        if edges is None: edges = {}

        for u, v in self.graph.edges(data=False):
            pt1 = self.latlon_to_pixel(u)
            pt2 = self.latlon_to_pixel(v)
            color_ = color

            if f'{u}-{v}' in edges.keys():
                color_ = value_to_rgb(1.0/edges[f'{u}-{v}'])
            cv2.line(self.base_img, pt1, pt2, color_, thickness)

        text = 'Edges: {}'.format(len(self.graph.edges()))
        if len(edges) > 0:
            text += '({})'.format(len(edges))
        cv2.putText(self.base_img, text, org=(60, 90), **text_kwargs)

    def draw_markers(self, markers=None):
        if markers is None: return
        for marker_name, node in markers.items():
            self.__draw_marker(node, marker_name)

    def __draw_marker(self, node, text, delta=(0,60)):
        pt = self.latlon_to_pixel(node)
        color = (128, 0, 128)
        cv2.circle(self.base_img, pt, 10, color, -1)
        cv2.putText(self.base_img, text,
                    org=(pt[0]+delta[0], pt[1] + delta[1]),
                    **text_kwargs)

    def draw_paths(self, paths):
        if paths is None: return

        count = 0
        for key, [end_node, cost, rew, path, done, *info] in paths.items():
            # color = make_random_color()
            color = colors[count % len(colors)]
            for i, u in enumerate(path[:-1]):
                v = path[i + 1]
                pt1 = self.latlon_to_pixel(u)
                pt2 = self.latlon_to_pixel(v)
                cv2.line(self.base_img, pt1, pt2, color, 4)

            self.__draw_marker(path[0], text='Start')
            if key == 'RL':
                self.__draw_marker(path[-1], text='End', delta=(0, 30))
            self.__draw_marker(end_node, text='Goal', delta=(-30, -30))

            text = '{:>15s} Cost:{:>5.2f}, Reward:{:>5.2f}, Length:{:>3d} ({})'.format(
                key, cost, rew, len(path), int(done)
            )
            new_kwargs = text_kwargs.copy()
            new_kwargs['color'] = color
            cv2.putText(self.base_img, text,
                        org=(self.size[0] - 960, 30 * count + 60),
                        **new_kwargs)

            if len(info) > 0:
                new_kwargs = text_kwargs.copy()
                new_kwargs['fontScale'] = 0.5
                new_kwargs['thickness'] = 1

                start_x, y = 60, 160
                for i, text in enumerate(info[0]):
                    if y >= self.size[1] - 30:
                        start_x, y = start_x + 180, 160
                    cv2.putText(self.base_img, text, org=(start_x, y), **new_kwargs)
                    y += 30
            count += 1

    def draw(self, name=None, show=True, **kwargs):
        self.base_img = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)

        self.draw_nodes(nodes=kwargs.get("nodes", None))
        self.draw_edges(edges=kwargs.get("edges", None))
        self.draw_paths(paths=kwargs.get("paths", None))
        self.draw_markers(markers=kwargs.get("markers", None))

        self.plot(name, show=show)

    def plot(self, name, show=False):
        if name is not None: cv2.imwrite(name + '.png', self.base_img)
        if show:
            cv2.imshow("Urban Road Network", self.base_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def close(self):
        self.graph = None
        self.base_img = None
        cv2.destroyAllWindows()


class DynamicRender:
    def __init__(self, env, height=720, padding=0.075, filename='video'):
        self.env = env
        self.graph = env.graph
        self.nodes = env.graph.nodes

        lats = [self.nodes[node]['y'] for node in self.nodes]
        lons = [self.nodes[node]['x'] for node in self.nodes]
        self.bounder = (min(lats), max(lats), min(lons), max(lons))

        delta_lat = (self.bounder[3] - self.bounder[2])
        delta_lon = (self.bounder[1] - self.bounder[0])
        width = int(height * delta_lat / delta_lon)

        self.padding = (int(width * padding), int(height * padding))
        self.size = (width, height)
        self.video_writer = cv2.VideoWriter(filename=filename+'.avi',
                                            fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                                            fps=10,
                                            frameSize=(width, height))
        self.base_img = None
        self.base_img_shadow = None
        self.build_base_image()

    def build_base_image(self):
        self.base_img = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8) * 255
        self.__draw_nodes(self.base_img)
        self.__draw_edges(self.base_img)
        self.base_img_shadow = self.base_img.copy()

    def latlon_to_pixel(self, lat, lon):
        min_lat, max_lat, min_lon, max_lon = self.bounder
        pad_w, pad_h = self.padding
        w_ = self.size[0] - pad_w * 2
        h_ = self.size[1] - pad_h * 2

        x = int((lon - min_lon) / (max_lon - min_lon) * w_)
        y = h_ - int((lat - min_lat) / (max_lat - min_lat) * h_)
        return x+pad_w, y+pad_h

    def __draw_nodes(self, img, nodes=None, vf:list=None):
        if vf is None: vf = []
        if nodes is None: nodes = self.nodes

        for i, u in enumerate(nodes):
            pt = self.latlon_to_pixel(nodes[u]['y'], nodes[u]['x'])
            color = (255, 0, 255) if u in vf else (0, 255, 0)
            cv2.circle(img, pt, 2, color, -1)

        if len(vf) > 0:
            text = 'Nodes: {}'.format(len(vf))
            cv2.putText(img, text, org=(60, 60), **text_kwargs)

    def __draw_edges(self, img, vf:list=None):
        if vf is None: vf = []

        count = 0
        for u, v, data in self.graph.edges(data=True):
            pt1 = self.latlon_to_pixel(self.nodes[u]['y'], self.nodes[u]['x'])
            pt2 = self.latlon_to_pixel(self.nodes[v]['y'], self.nodes[v]['x'])
            color = (0, 255, 0)
            if u in vf and v in vf:
                color = (255, 0, 255)
                count += 1

            if 'times' in data.keys():
                if data['times'] <= 0.9: color = (0, 0, 255)
            cv2.line(img, pt1, pt2, color, 1)

        if len(vf) > 0:
            text = 'Edges: {}'.format(count)
            cv2.putText(img, text, org=(60, 120), **text_kwargs)

    def __draw_markers(self, img, node, text):
        pt = self.latlon_to_pixel(self.nodes[node]['y'], self.nodes[node]['x'])
        cv2.circle(img, pt, 10, (0, 0, 255), -1)
        cv2.putText(img, text, org=(pt[0], pt[1] + 60), **text_kwargs)

    def __draw_path(self, img, path_info, color):
        path = path_info['path']
        for i, u in enumerate(path[:-1]):
            v = path[i + 1]
            pt1 = self.latlon_to_pixel(self.nodes[u]['y'], self.nodes[u]['x'])
            pt2 = self.latlon_to_pixel(self.nodes[v]['y'], self.nodes[v]['x'])
            cv2.line(img, pt1, pt2, color, 6)

        self.__draw_markers(img, path[0], text='Start')
        self.__draw_markers(img, path[-1], text='End')

        text = 'Cost: {} '.format(round(path_info['cost'], 2))
        text += '({}, {})'.format(len(path), path_info['done'])
        new_kwargs = text_kwargs.copy()
        new_kwargs['color'] = color
        cv2.putText(img, text, (self.size[0] - 500, 500), **new_kwargs)

    def draw(self, info=None, baseline=None, show=True, **kwargs):
        img = self.base_img_shadow.copy()
        self.__draw_edges(img)
        if baseline is not None:
            self.__draw_path(img, baseline, colors[0])

        # Current node
        u = self.env.cur_node
        pt = self.latlon_to_pixel(self.nodes[u]['y'], self.nodes[u]['x'])
        cv2.circle(img, pt, 10, (0, 0, 255), -1)
        cv2.putText(img, 'Here', (pt[0], pt[1] + 60), **text_kwargs)

        # Trajectory
        if self.env.last_node_idx is not None:
            v = self.env.last_node
            pt1 = self.latlon_to_pixel(self.nodes[u]['y'], self.nodes[u]['x'])
            pt2 = self.latlon_to_pixel(self.nodes[v]['y'], self.nodes[v]['x'])
            cv2.line(self.base_img_shadow, pt1, pt2, colors[1], 4)

        # W'_t
        if self.env.dfr.w_prime is not None:
            color = (128, 0, 128)
            for u, v in self.env.dfr.w_prime:
                pt1 = self.latlon_to_pixel(self.nodes[u]['y'], self.nodes[u]['x'])
                pt2 = self.latlon_to_pixel(self.nodes[v]['y'], self.nodes[v]['x'])
                cv2.circle(img, pt1, 5, color, -1)
                cv2.circle(img, pt2, 5, color, -1)
                cv2.line(img, pt1, pt2, color, 2)
        # W''_t
        if self.env.dfr.w_prime_prime is not None:
            color = (255, 0, 255)
            for u, v in self.env.dfr.w_prime_prime:
                pt1 = self.latlon_to_pixel(self.nodes[u]['y'], self.nodes[u]['x'])
                pt2 = self.latlon_to_pixel(self.nodes[v]['y'], self.nodes[v]['x'])
                cv2.circle(img, pt1, 5, color, -1)
                cv2.circle(img, pt2, 5, color, -1)
                cv2.line(img, pt1, pt2, color, 2)

        # Legend
        if info is not None:
            new_kwargs = text_kwargs.copy()
            new_kwargs['color'] = colors[1]
            cv2.putText(img, info['text'], (self.size[0] - 500, 60+500), **new_kwargs)
            if info['done']: self.build_base_image()

        self.video_writer.write(img)  # Write frame to video file
        if show:
            cv2.imshow("City Instant Delivery Simulation", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def close(self):
        self.video_writer.release()
        cv2.destroyAllWindows()
