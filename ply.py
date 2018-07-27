import sys
import numpy as np
from ply_exceptions import *

class PlyProperty:
    def __init__(self, type_, names):
        self.type_ = type_
        self.names = names

    def __repr__(self):
        msg = '(' + self.type_
        for name in self.names:
            msg += ', ' + name
        return msg + ')'

class PlyElement:
    def __init__(self, name, value, ply_properties = []):
        self.name = name
        self.value = value
        self.ply_properties = ply_properties

    def __repr__(self):
        return '(' + self.name + ', ' + self.value + ')'

def _read_header(file_path):
    DECODED_PLY_FILE_TYPE = 'ply'
    ACCEPTED_FORMATS = ['ascii', 'binary_little_endian']

    comments = []
    ply_elements = []
    ply_properties = []

    with open(file_path, 'rb') as file:
        # check for ply type in first line
        ply_file_type = file.readline().decode().rstrip('\n')
        if not ply_file_type == DECODED_PLY_FILE_TYPE:
            raise PlyParsingError('Did not find \'' + DECODED_PLY_FILE_TYPE + '\' file type in first line of file')
        # read format
        ply_format = file.readline().split()[1].decode()
        if not ply_format in ACCEPTED_FORMATS:
            raise PlyParsingError('Format found in second line (' + ply_format + ') is not accepted. '
                                                                                 'Accepted formats are: '
                                  + str(ACCEPTED_FORMATS))
        # read the rest of header
        line_header = file.readline()
        element_name_previous = ''
        element_value = ''
        vertex_begin = 3
        num_faces = -1
        while not line_header == b'end_header\n':
            # count the lines to get the header's last line
            vertex_begin += 1
            # read in comments
            if b'comment' in line_header:
                comment = line_header.decode().rstrip('\n')[len('comment') + 1:]
                comments.append(comment)
            if b'element' in line_header:
                element_name = line_header.split()[1].decode().strip('\n')
                element_value = line_header.split()[2].decode().strip('\n')
                if element_name == 'vertex':
                    num_vertices = int(element_value)
                elif element_name == 'face':
                    num_faces = int(element_value)
                if not element_name_previous == '':
                    ply_elements.append(PlyElement(element_name_previous, element_value_previous, ply_properties))
                    ply_properties = []
                element_name_previous = element_name
                element_value_previous = element_value
            if b'property' in line_header:
                ply_property_type = line_header.decode().strip('\n').split()[1]
                ply_property_values = line_header.decode().strip('\n').split()[2:]
                ply_properties.append(PlyProperty(ply_property_type, ply_property_values))
            line_header = file.readline()
        # deal with properties read for last element
        if len(ply_properties) > 0:
            ply_elements.append(PlyElement(element_name_previous, element_value_previous, ply_properties))
        faces_begin = vertex_begin + num_vertices
        return ply_format, comments, ply_elements, vertex_begin, faces_begin

def _read_vertices_ply(file_path, ply_format, ply_elements, vertex_begin):
    """Reads vertices from a PLY file."""
    # get vertex element
    element_vertex = ''
    element_face = ''
    for ply_element in ply_elements:
        if ply_element.name == 'vertex':
            element_vertex = ply_element
        if ply_element.name == 'face':
            element_face = ply_element
    if element_vertex == '':
        raise PlyParsingError('Could not find element vertex')
    # get vertex property types
    vertex_prop_types = []
    vertex_prop_names = []
    for ply_property in element_vertex.ply_properties:
        if ply_property.type_ == 'uchar':
            vertex_prop_types.append('int')
        else:
            vertex_prop_types.append(ply_property.type_)
        vertex_prop_names.append(ply_property.names[0])
    vertices = np.array([]).astype(float)
    vertices_normals = np.array([]).astype(float)
    segm_ixs = np.array([]).astype(int)
    vertices_colors = np.array([]).astype(int)
    if ply_format == 'ascii':
        with open(file_path) as file:
            file_vertex_list = list(file)[vertex_begin:vertex_begin+int(element_vertex.value)]
        vertex_matrix = np.array([vertex_line.strip().split() for vertex_line in file_vertex_list])
        vertex_dict = {}
        for vertex_prop_name in vertex_prop_names:
            ix_prop_name = vertex_prop_names.index(vertex_prop_name)
            vertex_dict[vertex_prop_name] = vertex_matrix[:, ix_prop_name].astype(vertex_prop_types[ix_prop_name])
        if 'x' in vertex_dict and 'y' in vertex_dict and 'z' in vertex_dict:
            vertices = np.vstack((vertex_dict['x'], vertex_dict['y']))
            vertices = np.vstack((vertices, vertex_dict['z'])).T
        else:
            raise PlyParsingError('Could not parse vertex property for points.')
        if 'nx' in vertex_dict and 'ny' in vertex_dict and 'nz' in vertex_dict:
            vertices_normals = np.vstack((vertex_dict['nx'], vertex_dict['ny']))
            vertices_normals = np.vstack((vertices_normals, vertex_dict['nz'])).T
        if 'red' in vertex_dict and 'green' in vertex_dict and 'blue' in vertex_dict:
            vertices_colors = np.vstack((vertex_dict['red'], vertex_dict['green']))
            vertices_colors = np.vstack((vertices_colors, vertex_dict['blue']))
            if 'alpha' in vertex_dict:
                vertices_colors = np.vstack((vertices_colors, vertex_dict['alpha']))
            vertices_colors = vertices_colors.T
        if 'segm' in vertex_dict:
            segm_ixs = np.array(vertex_dict['segm']).reshape((vertices.shape[0], 1))
    return vertices, vertices_normals, segm_ixs, vertices_colors

def _read_faces(file_path, ply_format, faces_begin):
    """Reads vertices from a PLY file."""
    faces = np.array([])
    face_normals = np.array([])
    if ply_format == 'ascii':
        with open(file_path) as file:
            file_face_list = list(file)[faces_begin:]
            faces = np.array([face_line.strip().split() for face_line in file_face_list]).astype(int)
            # ignore first column that indicates number of faces
            faces = faces[:, 1:]
    return faces, face_normals

def read_from_filepath(file_path):
    ply_format, comments, ply_elements, vertex_begin, faces_begin = _read_header(file_path)
    vertices, vertices_normals, segm_ixs, vertices_colors = _read_vertices_ply(file_path, ply_format, ply_elements,
                                                                               vertex_begin)
    faces, face_normals = _read_faces(file_path, ply_format, faces_begin)
    return comments, vertices, vertices_normals, segm_ixs, vertices_colors, faces, face_normals

def write_to_filepath(comments, vertices, vertices_normals, segm_ixs, vertices_colors, faces, faces_normals, file_path):
    try:
        file_id = open(file_path, 'w+')
    except:
        print('Could not open file: ' + file_path)
        sys.exit(1)
    file_id.write('ply')
    file_id.write('\n')
    file_id.write('format ascii 1.0')
    file_id.write('\n')
    # comments
    for ply_comment in comments:
        file_id.write('comment ' + ply_comment)
        file_id.write('\n')
    # vertices
    if vertices is None or len(vertices) < 1:
        raise PlyWritingError('Point cloud has no vertices')
    file_id.write('element ' + 'vertex ' + str(len(vertices)))
    file_id.write('\n')
    file_id.write('property float x')
    file_id.write('\n')
    file_id.write('property float y')
    file_id.write('\n')
    file_id.write('property float z')
    file_id.write('\n')
    vertex_matrix = vertices
    vertex_matrix_col_types = ['float', 'float', 'float']
    # vertex normals
    if not vertices_normals is None and len(vertices_normals) > 0:
        file_id.write('property float nx')
        file_id.write('\n')
        file_id.write('property float ny')
        file_id.write('\n')
        file_id.write('property float nz')
        file_id.write('\n')
        vertex_matrix = np.hstack((vertex_matrix, vertices_normals))
        vertex_matrix_col_types += ['float', 'float', 'float']
    # segmentation indices
    if not segm_ixs is None and len(segm_ixs) > 0:
        file_id.write('property int segm')
        file_id.write('\n')
        vertex_matrix = np.hstack((vertex_matrix, segm_ixs))
        vertex_matrix_col_types.append('int')
    # colors
    if not vertices_colors is None and len(vertices_colors) > 0:
        file_id.write('property uchar red')
        file_id.write('\n')
        file_id.write('property uchar green')
        file_id.write('\n')
        file_id.write('property uchar blue')
        file_id.write('\n')
        vertex_matrix_col_types += ['int', 'int', 'int']
        if vertices_colors.shape[1] == 4:
            file_id.write('property uchar alpha')
            file_id.write('\n')
            vertex_matrix_col_types.append('int')
        vertex_matrix = np.hstack((vertex_matrix, vertices_colors))
    if faces is not None and faces.shape[0] > 0:
        file_id.write('element face ' + '{}'.format(faces.shape[0]))
        file_id.write('\n')
        file_id.write('property list uchar int vertex_indices')
        file_id.write('\n')
    # close header
    file_id.write('end_header')
    # write available vertex information (vertices, normals, segmentation and colors)
    for i in range(vertex_matrix.shape[0]):
        line_vertex = '{}'.format(vertex_matrix[i, 0].astype(vertex_matrix_col_types[0]))
        for j in range(vertex_matrix.shape[1]-1):
            line_vertex += ' ' + '{}'.format(vertex_matrix[i, j+1].astype(vertex_matrix_col_types[j+1]))
        file_id.write('\n')
        file_id.write(line_vertex)
    file_id.write('\n')
    # write faces
    if faces.shape[0] > 0:
        n_face_vertices = faces.shape[1]
        for i in range(faces.shape[0]):
            line_face = '{}'.format(n_face_vertices)
            for j in range(faces.shape[1]):
                line_face += ' ' + '{}'.format(faces[i, j].astype(int))
            file_id.write(line_face)
            file_id.write('\n')
    # close file
    file_id.close()