bl_info = {
    "name": "CuFloda - CUDA Fluid Dynamics",
    "author": "CuFloda Team",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > CuFloda",
    "description": "GPU-accelerated fluid simulation using Lattice Boltzmann Method",
    "category": "Physics",
}

import bpy
from bpy.props import IntProperty, FloatProperty, BoolProperty, PointerProperty
from bpy.types import Panel, Operator, PropertyGroup

class CuFlodaProperties(PropertyGroup):
    width: IntProperty(
        name="Width",
        description="Simulation grid width",
        default=256,
        min=64,
        max=1024
    )
    
    height: IntProperty(
        name="Height",
        description="Simulation grid height",
        default=256,
        min=64,
        max=1024
    )
    
    viscosity: FloatProperty(
        name="Viscosity",
        description="Fluid viscosity",
        default=0.02,
        min=0.001,
        max=0.2
    )
    
    steps_per_frame: IntProperty(
        name="Steps per Frame",
        description="Simulation steps per frame",
        default=10,
        min=1,
        max=100
    )
    
    initial_velocity_x: FloatProperty(
        name="Initial Velocity X",
        description="Initial velocity in X direction",
        default=0.1,
        min=-1.0,
        max=1.0
    )
    
    initial_velocity_y: FloatProperty(
        name="Initial Velocity Y",
        description="Initial velocity in Y direction",
        default=0.0,
        min=-1.0,
        max=1.0
    )
    
    obstacle_object: PointerProperty(
        name="Obstacle Object",
        description="Blender object to use as obstacle",
        type=bpy.types.Object
    )

class CuFlodaPanel(Panel):
    bl_label = "CuFloda Fluid Simulation"
    bl_idname = "VIEW3D_PT_cufloda"
    bl_space_type = 'VIEW3D'
    bl_region_type = 'UI'
    bl_category = 'CuFloda'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.cufloda_props
        
        layout.prop(props, "width")
        layout.prop(props, "height")
        layout.prop(props, "viscosity")
        layout.prop(props, "steps_per_frame")
        layout.prop(props, "initial_velocity_x")
        layout.prop(props, "initial_velocity_y")
        layout.prop(props, "obstacle_object")
        
        layout.separator()
        
        layout.operator("cufloda.initialize_simulation")
        layout.operator("cufloda.set_obstacles")
        layout.operator("cufloda.run_step")
        layout.operator("cufloda.run_simulation")
        layout.operator("cufloda.export_particles")

class CuFlodaInitializeSimulation(Operator):
    bl_idname = "cufloda.initialize_simulation"
    bl_label = "Initialize Simulation"
    
    def execute(self, context):
        try:
            import cufloda
            props = context.scene.cufloda_props
            
            sim = cufloda.LBM2D(props.width, props.height, props.viscosity)
            sim.initialize(1.0, props.initial_velocity_x, props.initial_velocity_y)
            
            context.scene.cufloda_simulation = sim
            self.report({'INFO'}, f"Simulation initialized: {props.width}x{props.height}")
            return {'FINISHED'}
        except ImportError:
            self.report({'ERROR'}, "CuFloda module not found. Install with: pip install -e .")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to initialize: {str(e)}")
            return {'CANCELLED'}

class CuFlodaSetObstacles(Operator):
    bl_idname = "cufloda.set_obstacles"
    bl_label = "Set Obstacles"
    
    def execute(self, context):
        sim = getattr(context.scene, 'cufloda_simulation', None)
        if sim is None:
            self.report({'ERROR'}, "No active simulation. Initialize first.")
            return {'CANCELLED'}
        
        props = context.scene.cufloda_props
        obstacle_obj = props.obstacle_object
        
        if obstacle_obj is None:
            self.report({'WARNING'}, "No obstacle object selected")
            return {'CANCELLED'}
        
        import numpy as np
        
        # Create obstacle mask from object
        obstacles = np.zeros((props.height, props.width), dtype=bool)
        
        # Simple bounding box obstacle for now
        # In future, use actual mesh geometry
        if obstacle_obj.type == 'MESH':
            bbox = obstacle_obj.bound_box
            min_x = min(v[0] for v in bbox)
            max_x = max(v[0] for v in bbox)
            min_y = min(v[1] for v in bbox)
            max_y = max(v[1] for v in bbox)
            
            # Convert to grid coordinates
            grid_min_x = int(min_x)
            grid_max_x = int(max_x)
            grid_min_y = int(min_y)
            grid_max_y = int(max_y)
            
            # Clamp to grid bounds
            grid_min_x = max(0, grid_min_x)
            grid_max_x = min(props.width, grid_max_x)
            grid_min_y = max(0, grid_min_y)
            grid_max_y = min(props.height, grid_max_y)
            
            obstacles[grid_min_y:grid_max_y, grid_min_x:grid_max_x] = True
        
        sim.set_obstacles(obstacles)
        self.report({'INFO'}, f"Obstacles set from {obstacle_obj.name}")
        return {'FINISHED'}

class CuFlodaRunStep(Operator):
    bl_idname = "cufloda.run_step"
    bl_label = "Run Single Step"
    
    def execute(self, context):
        sim = getattr(context.scene, 'cufloda_simulation', None)
        if sim is None:
            self.report({'ERROR'}, "No active simulation. Initialize first.")
            return {'CANCELLED'}
        
        sim.step()
        self.report({'INFO'}, "Step completed")
        return {'FINISHED'}

class CuFlodaRunSimulation(Operator):
    bl_idname = "cufloda.run_simulation"
    bl_label = "Run Simulation"
    
    def execute(self, context):
        sim = getattr(context.scene, 'cufloda_simulation', None)
        if sim is None:
            self.report({'ERROR'}, "No active simulation. Initialize first.")
            return {'CANCELLED'}
        
        props = context.scene.cufloda_props
        sim.run(props.steps_per_frame)
        self.report({'INFO'}, f"Ran {props.steps_per_frame} steps")
        return {'FINISHED'}

class CuFlodaExportParticles(Operator):
    bl_idname = "cufloda.export_particles"
    bl_label = "Export to Particles"
    
    def execute(self, context):
        sim = getattr(context.scene, 'cufloda_simulation', None)
        if sim is None:
            self.report({'ERROR'}, "No active simulation. Initialize first.")
            return {'CANCELLED'}
        
        density = sim.get_density()
        velocity = sim.get_velocity()
        
        import numpy as np
        
        # Create particles from high-density regions
        threshold = 1.01
        y_indices, x_indices = np.where(density > threshold)
        
        if len(x_indices) == 0:
            self.report({'WARNING'}, "No particles above threshold")
            return {'CANCELLED'}
        
        # Create mesh from particles
        mesh = bpy.data.meshes.new("CuFlodaParticles")
        obj = bpy.data.objects.new("CuFlodaParticles", mesh)
        context.collection.objects.link(obj)
        
        verts = []
        for x, y in zip(x_indices, y_indices):
            verts.append((float(x), float(y), 0.0))
        
        edges = []
        faces = []
        
        mesh.from_pydata(verts, edges, faces)
        mesh.update()
        
        self.report({'INFO'}, f"Exported {len(verts)} particles")
        return {'FINISHED'}

classes = (
    CuFlodaProperties,
    CuFlodaPanel,
    CuFlodaInitializeSimulation,
    CuFlodaSetObstacles,
    CuFlodaRunStep,
    CuFlodaRunSimulation,
    CuFlodaExportParticles,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.cufloda_props = PointerProperty(type=CuFlodaProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.cufloda_props

if __name__ == "__main__":
    register()
