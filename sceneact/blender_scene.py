from functools import singledispatch
from .kernel import Brick, Orb, Bar
from .composition import Scene


@singledispatch
def _mesh(body):
    raise TypeError(f'No mesh emitter for {type(body).__name__}')


@_mesh.register
def _(body: Brick):
    import bpy
    bpy.ops.mesh.primitive_cube_add(size=1, location=body.origin)
    mesh=bpy.context.object
    mesh.scale=body.dimensions
    return mesh


@_mesh.register
def _(body: Orb):
    import bpy
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=body.radius, location=body.origin)
    return bpy.context.object


@_mesh.register
def _(body: Bar):
    import bpy
    from mathutils import Vector
    start,end=Vector(body.start),Vector(body.end)
    heading=end-start
    if heading.length<=1e-9: raise ValueError('Cannot mesh a rod with coincident endpoints')
    bpy.ops.mesh.primitive_cylinder_add(vertices=32,radius=body.radius,depth=heading.length,location=(start+end)/2)
    result=bpy.context.object
    result.rotation_euler=heading.to_track_quat('Z','Y').to_euler()
    return result


class SceneCompiler:
    def __init__(self):
        self.materials={}

    def material(self,finish):
        import bpy
        if finish not in self.materials:
            material=bpy.data.materials.new(f'SceneActSurface_{len(self.materials)}')
            material.use_nodes=True
            material.diffuse_color=finish.rgba
            inputs=material.node_tree.nodes.get('Principled BSDF').inputs
            for key,value in {'Base Color':finish.rgba,'Alpha':finish.rgba[3],'Roughness':.65}.items():
                inputs[key].default_value=value
            self.materials[finish]=material
        return self.materials[finish]

    def collapse(self,objects):
        import bpy
        if not objects: raise ValueError('Empty Boolean operand')
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects: obj.select_set(True)
        bpy.context.view_layer.objects.active=objects[0]
        if len(objects)>1: bpy.ops.object.join()
        return objects[0]

    def emit(self,node,parent=None,slot=None):
        import bpy
        from mathutils import Matrix
        marker=bpy.data.objects.new(slot or node.name,None)
        bpy.context.collection.objects.link(marker)
        marker.parent=parent
        marker.matrix_world=Matrix(node.world_frame.tolist())
        objects=[]
        if node.solid is not None:
            obj=_mesh(node.solid)
            bpy.context.view_layer.update()
            world=Matrix(node.geometry_frame.tolist()) @ obj.matrix_world.copy()
            obj.parent=marker
            obj.matrix_world=world
            obj.name=node.name
            obj.data.materials.append(self.material(node.finish))
            objects.append(obj)
        operands=[self.emit(child,marker,name) for name,child in node.members.items()]
        if node.operation and operands:
            meshes=[self.collapse(group) for group in operands]
            result=meshes[0]
            mode={'union':'UNION','intersection':'INTERSECT','difference':'DIFFERENCE'}[node.operation]
            for tool in meshes[1:]:
                bpy.context.view_layer.objects.active=result
                modifier=result.modifiers.new('SceneActCSG','BOOLEAN')
                modifier.operation=mode
                modifier.object=tool
                modifier.solver='EXACT'
                bpy.ops.object.modifier_apply(modifier=modifier.name)
                bpy.data.objects.remove(tool,do_unlink=True)
            objects.append(result)
        else:
            objects.extend(obj for group in operands for obj in group)
        return objects


def realize(scene, clear=True):
    import bpy
    if clear: bpy.ops.wm.read_factory_settings(use_empty=True)
    for node in scene.traverse():
        if isinstance(node,Scene): node.settle()
    return SceneCompiler().emit(scene)


def _aimed_object(name,data,position,target):
    import bpy
    from mathutils import Vector
    result=bpy.data.objects.new(name,data)
    bpy.context.collection.objects.link(result)
    result.location=position
    ray=Vector(target)-Vector(position)
    if ray.length<=1e-12: raise ValueError('View direction must have positive length')
    result.rotation_euler=ray.to_track_quat('-Z','Y').to_euler()
    return result


def set_view(position,target,lens=35,sensor=36,shift=(0,0)):
    import bpy
    data=bpy.data.cameras.new('SceneActView')
    data.lens,data.sensor_width=lens,sensor
    data.shift_x,data.shift_y=shift
    result=_aimed_object('SceneActView',data,position,target)
    bpy.context.scene.camera=result
    return result


def area_emitter(position,power=500,size=5,target=(0,0,0),color=(1,1,1)):
    import bpy
    data=bpy.data.lights.new('SceneActArea','AREA')
    data.energy,data.size,data.color=power,size,color
    return _aimed_object('SceneActArea',data,position,target)
