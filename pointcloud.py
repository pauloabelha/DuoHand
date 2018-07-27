import ply as ply
import numpy as np

class PointCloud:
    KNOWN_FILETYPES = ['ply']
    BAMBA_COMMENT = 'Processed using bamba (github.com/pauloabelha/bamba)'

    def __init__(self, vertices, vertices_normals=[], segm_ixs=[],
                 vertices_colors=[], faces=[], faces_normals=[],
                 comments=[], file_path=""):
        vertices = np.array(vertices)
        vertices_normals = np.array(vertices_normals)
        segm_ixs = np.array(segm_ixs).astype(int)
        vertices_colors = np.array(vertices_colors).astype(int)
        faces = np.array(faces)
        faces_normals = np.array(faces_normals)
        self.file_path = file_path
        self.comments = comments
        assert(vertices.shape[0] > 0 and vertices.shape[1] == 3)
        self.vertices = vertices
        if vertices_normals.shape[0] > 0:
            assert(vertices_normals.shape[1] == 3)
        self.vertices_normals = vertices_normals
        self.segm_ixs = segm_ixs
        if vertices_colors.shape[0] > 0:
            assert(vertices_colors.shape[1] == 3 or vertices_colors.shape[1] == 4)
        self.vertices_colors = vertices_colors
        if faces.shape[0] > 0:
            assert(faces.shape[1] == 3 or faces.shape[1] == 4)
        self.faces = faces
        if faces_normals.shape[0] > 0:
            assert(faces_normals.shape[1] == 3)
        self.faces_normals = faces_normals

    @classmethod
    def from_file(self, file_path):
        file_type = file_path.split('.')[1][-3:]
        if not file_type in self.KNOWN_FILETYPES:
            raise BaseException('File format ' + file_type + ' is not supported.')
        if file_type == 'ply':
            comments, vertices, vertices_normals, segm_ixs, vertices_colors, faces, faces_normals = \
                ply.read_from_filepath(file_path)
            return PointCloud(vertices, vertices_normals, segm_ixs,
                              vertices_colors, faces, faces_normals, comments, file_path)
        return None


    def write(self, file_path):
        # add BAMBA comment
        self.comments.append(self.BAMBA_COMMENT)
        file_type = file_path.split('.')[1][-3:]
        if not file_type in self.KNOWN_FILETYPES:
            raise BaseException('File format ' + file_type + ' is not supported.')
        if file_type == 'ply':
            ply.write_to_filepath(self.comments, self.vertices, self.vertices_normals, self.segm_ixs,
                                  self.vertices_colors, self.faces, self.faces_normals, file_path)

    def centralize(self):
        for i in range(3):
            self.vertices[:, i] -= np.mean(self.vertices[:, i])


    def bound_box(self):
        bbox = np.zeros((6,))
        for i in range(3):
            bbox[i] = np.min(self.vertices[:, i])
        for i in range(3):
            bbox[i+3] = np.max(self.vertices[:, i])
        return bbox

