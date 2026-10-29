#include "raylib.h"
#define RAYMATH_STATIC_INLINE
#include "raymath.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
#include "tinyfiledialogs.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define APP_TITLE "S-Stream | Wind Tunnel"
#define SIDEBAR_WIDTH 292.0f

typedef struct {
    Model model;
    bool loaded;
    char path[1024];
    Vector3 center;
    float radius;
    float scale;
    float rotation[3];
} SceneModel;

static Color bg = {18, 20, 25, 255};
static Color panel = {27, 30, 37, 255};
static Color line = {48, 53, 63, 255};
static Color muted = {151, 160, 175, 255};
static Color accent = {79, 171, 225, 255};

static bool LoadSceneModel(SceneModel *scene, const char *path, char *error, size_t errorSize) {
    if (!FileExists(path)) {
        snprintf(error, errorSize, "File not found: %s", path);
        return false;
    }
    Model candidate = LoadModel(path);
    if (candidate.meshCount <= 0 || candidate.meshes == NULL) {
        if (candidate.meshes != NULL) UnloadModel(candidate);
        snprintf(error, errorSize, "Could not load this model. Try a valid OBJ file.");
        return false;
    }
    BoundingBox bounds = GetModelBoundingBox(candidate);
    Vector3 center = Vector3Scale(Vector3Add(bounds.min, bounds.max), 0.5f);
    Vector3 size = Vector3Subtract(bounds.max, bounds.min);
    float radius = 0.5f * sqrtf(size.x*size.x + size.y*size.y + size.z*size.z);
    if (!(radius > 0.00001f) || !isfinite(radius)) {
        UnloadModel(candidate);
        snprintf(error, errorSize, "Model has empty or invalid dimensions.");
        return false;
    }
    if (scene->loaded) UnloadModel(scene->model);
    scene->model = candidate;
    scene->loaded = true;
    scene->center = center;
    scene->radius = radius;
    scene->scale = 1.0f;
    scene->rotation[0] = scene->rotation[1] = scene->rotation[2] = 0.0f;
    strncpy(scene->path, path, sizeof(scene->path) - 1);
    scene->path[sizeof(scene->path) - 1] = '\0';
    error[0] = '\0';
    return true;
}

static void ResetCamera(Camera3D *camera, float yaw, float pitch, float distance) {
    camera->target = (Vector3){0, 0, 0};
    camera->position = (Vector3){distance*sinf(yaw)*cosf(pitch), distance*sinf(pitch), distance*cosf(yaw)*cosf(pitch)};
}

static void OrbitCamera(Camera3D *camera, float yaw, float pitch, float distance) {
    camera->position = Vector3Add(camera->target,
        (Vector3){distance*sinf(yaw)*cosf(pitch), distance*sinf(pitch), distance*cosf(yaw)*cosf(pitch)});
}

static void DrawInfiniteGrid(Vector3 cameraPos, float spacing, int divisions) {
    float range = spacing * divisions;
    float offsetX = fmodf(cameraPos.x, spacing);
    float offsetZ = fmodf(cameraPos.z, spacing);
    
    Color gridColor = {60, 65, 75, 150};
    Color axisColor = {100, 105, 115, 200};
    
    for (int i = -divisions; i <= divisions; i++) {
        float x = cameraPos.x - offsetX + i * spacing;
        float z = cameraPos.z - offsetZ + i * spacing;
        
        // X-axis lines (parallel to X)
        Vector3 start = {cameraPos.x - range, 0, z};
        Vector3 end = {cameraPos.x + range, 0, z};
        DrawLine3D(start, end, (i == 0) ? axisColor : gridColor);
        
        // Z-axis lines (parallel to Z)
        start = (Vector3){x, 0, cameraPos.z - range};
        end = (Vector3){x, 0, cameraPos.z + range};
        DrawLine3D(start, end, (i == 0) ? axisColor : gridColor);
    }
}

int main(void) {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(1440, 900, APP_TITLE);
    SetTargetFPS(60);
    GuiSetStyle(DEFAULT, TEXT_SIZE, 16);
    GuiSetStyle(DEFAULT, BACKGROUND_COLOR, ColorToInt(panel));
    GuiSetStyle(DEFAULT, BORDER_COLOR_NORMAL, ColorToInt(line));
    GuiSetStyle(DEFAULT, BASE_COLOR_NORMAL, ColorToInt(panel));
    GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, ColorToInt(RAYWHITE));
    GuiSetStyle(DEFAULT, BASE_COLOR_FOCUSED, ColorToInt((Color){42, 51, 63, 255}));
    GuiSetStyle(DEFAULT, BASE_COLOR_PRESSED, ColorToInt(accent));

    SceneModel scene = {0};
    Camera3D camera = {0};
    camera.position = (Vector3){4.5f, 3.2f, 5.5f};
    camera.target = (Vector3){0, 0, 0};
    camera.up = (Vector3){0, 1, 0};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    float yaw = 0.69f, pitch = 0.42f, distance = 8.5f;
    ResetCamera(&camera, yaw, pitch, distance);
    bool showGrid = true, showAxes = true, showAbout = false;
    char error[512] = "";
    char status[256] = "Import an OBJ model to begin.";
    char defaultModel[1024];
    snprintf(defaultModel, sizeof(defaultModel), "%sassets/3d-files/car-model.obj", GetApplicationDirectory());
    LoadSceneModel(&scene, defaultModel, error, sizeof(error));
    if (scene.loaded) {
        distance = scene.radius*2.7f;
        ResetCamera(&camera, yaw, pitch, distance);
        snprintf(status, sizeof(status), "Loaded %s", GetFileName(scene.path));
    }

    while (!WindowShouldClose()) {
        float sw = (float)GetScreenWidth(), sh = (float)GetScreenHeight();
        float sidebar = fminf(SIDEBAR_WIDTH, sw * 0.38f);
        Rectangle viewport = {sidebar, 0, sw-sidebar, sh-30};
        Vector2 mouse = GetMousePosition();
        bool inView = CheckCollisionPointRec(mouse, viewport);
        if (scene.loaded) {
            if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && inView && !GuiIsLocked()) {
                Vector2 delta = GetMouseDelta(); yaw -= delta.x*0.006f; pitch += delta.y*0.006f;
                if (pitch > 1.52f) pitch = 1.52f;
                if (pitch < -1.52f) pitch = -1.52f;
                OrbitCamera(&camera, yaw, pitch, distance);
            }
            if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE) && inView) {
                Vector2 delta = GetMouseDelta();
                Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
                Vector3 right = Vector3Normalize(Vector3CrossProduct(forward, camera.up));
                Vector3 up = Vector3CrossProduct(right, forward);
                float factor = distance * 0.0025f;
                camera.target = Vector3Add(camera.target, Vector3Add(Vector3Scale(right, -delta.x*factor), Vector3Scale(up, delta.y*factor)));
                camera.position = Vector3Add(camera.position, Vector3Add(Vector3Scale(right, -delta.x*factor), Vector3Scale(up, delta.y*factor)));
            }
            if (inView && GetMouseWheelMove() != 0.0f) {
                distance *= 1.0f - GetMouseWheelMove()*0.1f;
                if (distance < scene.radius*1.1f) distance = scene.radius*1.1f;
                if (distance > scene.radius*100.0f) distance = scene.radius*100.0f;
                OrbitCamera(&camera, yaw, pitch, distance);
            }
        }

        BeginDrawing();
        ClearBackground(bg);

        DrawRectangle(0, 0, (int)sidebar, (int)sh-30, panel);
        DrawLine((int)sidebar, 0, (int)sidebar, (int)sh-30, line);
        int y=20;
        DrawText("S-STREAM", 20, y, 18, RAYWHITE); y+=25;
        DrawText("WIND TUNNEL", 20, y, 13, muted); y+=35;
        DrawText("SCENE", 20,y,12,muted); y+=30;
        DrawText(scene.loaded ? GetFileName(scene.path) : "No model loaded",20,y,18,RAYWHITE); y+=28;
        if (GuiButton((Rectangle){20,(float)y,sidebar-40,38}, "Import OBJ...")) {
            const char *filters[]={"*.obj"};
            const char *selected=tinyfd_openFileDialog("Import OBJ model","",1,filters,"Wavefront OBJ",0);
            if(selected) {
                if(!LoadSceneModel(&scene,selected,error,sizeof(error))) snprintf(status,sizeof(status),"%s",error);
                else {distance=scene.radius*2.7f; yaw=0.69f; pitch=0.42f; ResetCamera(&camera,yaw,pitch,distance); snprintf(status,sizeof(status),"Loaded %s",GetFileName(scene.path));}
            }
        }
        y+=53;
        if (GuiButton((Rectangle){20,(float)y,sidebar-40,32}, "Fit model to view")) {
            if(scene.loaded){distance=scene.radius*2.7f;camera.target=(Vector3){0,0,0};ResetCamera(&camera,yaw,pitch,distance);}
        }
        y+=43;
        DrawLine(20,y,(int)sidebar-20,y,line); y+=18;
        DrawText("TRANSFORM",20,y,12,muted); y+=26;
        DrawText("Scale",20,y+5,15,RAYWHITE);
        GuiSlider((Rectangle){105,(float)y,sidebar-130,24},NULL,TextFormat("%.2fx",scene.scale),&scene.scale,0.1f,5.0f); y+=37;
        const char *axis[] = {"Rotate X","Rotate Y","Rotate Z"};
        for(int i=0;i<3;i++){DrawText(axis[i],20,y+5,14,RAYWHITE); GuiSlider((Rectangle){105,(float)y,sidebar-130,24},NULL,TextFormat("%.0f°",scene.rotation[i]),&scene.rotation[i],-180,180); y+=34;}
        if(scene.loaded){BoundingBox b=GetModelBoundingBox(scene.model); Vector3 size=Vector3Subtract(b.max,b.min); y+=8; DrawText("MODEL BOUNDS",20,y,12,muted); y+=22; DrawText(TextFormat("X %.3f   Y %.3f   Z %.3f",size.x,size.y,size.z),20,y,13,RAYWHITE); y+=30;}
        DrawLine(20,y,(int)sidebar-20,y,line); y+=18;
        DrawText("DISPLAY",20,y,12,muted); y+=27;
        GuiCheckBox((Rectangle){20,(float)y,20,20},"Ground grid",&showGrid); y+=30;
        GuiCheckBox((Rectangle){20,(float)y,20,20},"XYZ axes",&showAxes); y+=39;
        DrawLine(20,y,(int)sidebar-20,y,line); y+=18;
        DrawText("SIMULATION",20,y,12,muted); y+=25;
        DrawText("Solver not connected yet",20,y,13,muted); y+=28;
        DrawText("Wind speed",20,y,14,muted); GuiDisable(); GuiSlider((Rectangle){120,(float)y,sidebar-145,22},NULL,"1.0 m/s",&(float){1.0f},0,50); GuiEnable(); y+=40;
        DrawLine(20,y,(int)sidebar-20,y,line); y+=18;
        if (GuiButton((Rectangle){20,(float)y,sidebar-40,32}, "Help (F1)")) showAbout=true;
        if (IsKeyPressed(KEY_F1)) showAbout=true;

        BeginScissorMode((int)viewport.x,(int)viewport.y,(int)viewport.width,(int)viewport.height);
        DrawRectangleRec(viewport,bg);
        if(scene.loaded){
            BeginMode3D(camera);
            if(showGrid) DrawInfiniteGrid(camera.position, 0.5f, 24);
            if(showAxes){DrawLine3D((Vector3){0,0,0},(Vector3){2,0,0},RED);DrawLine3D((Vector3){0,0,0},(Vector3){0,2,0},GREEN);DrawLine3D((Vector3){0,0,0},(Vector3){0,0,2},BLUE);}
            Matrix rotation=MatrixRotateXYZ((Vector3){DEG2RAD*scene.rotation[0],DEG2RAD*scene.rotation[1],DEG2RAD*scene.rotation[2]});
            Matrix centering=MatrixTranslate(-scene.center.x,-scene.center.y,-scene.center.z);
            scene.model.transform=MatrixMultiply(MatrixMultiply(MatrixScale(scene.scale,scene.scale,scene.scale),rotation),centering);
            DrawModel(scene.model,(Vector3){0,0,0},1.0f,WHITE);
            EndMode3D();
            DrawText("LMB orbit   MMB pan   Wheel zoom",(int)viewport.x+18,(int)sh-58,13,muted);
            // Blender-style overlay info
            int infoX = (int)viewport.x + 18;
            int infoY = (int)viewport.y + 18;
            DrawText(TextFormat("FPS: %d", GetFPS()), infoX, infoY, 14, (Color){180, 185, 195, 255});
            DrawText(TextFormat("Cam: %.1f, %.1f, %.1f", camera.position.x, camera.position.y, camera.position.z), infoX, infoY + 20, 13, (Color){140, 145, 155, 255});
            if(scene.loaded){
                int totalVerts = 0, totalTris = 0;
                for(int i = 0; i < scene.model.meshCount; i++){
                    totalVerts += scene.model.meshes[i].vertexCount;
                    totalTris += scene.model.meshes[i].triangleCount;
                }
                DrawText(TextFormat("Verts: %d | Tris: %d", totalVerts, totalTris), infoX, infoY + 38, 13, (Color){140, 145, 155, 255});
            }
        } else DrawText("Import an OBJ model to inspect it",(int)viewport.x+32,(int)viewport.y+32,20,muted);
        EndScissorMode();
        DrawRectangle(0,(int)sh-30,(int)sw,30,panel); DrawLine(0,(int)sh-30,(int)sw,(int)sh-30,line);
        DrawText(status,16,(int)sh-22,13,error[0]? (Color){255,130,120,255}:muted);
        if(showAbout){ if(GuiMessageBox((Rectangle){sw/2-170,sh/2-75,340,150},"About S-Stream","3D model inspection shell. Simulation is not connected yet.","Close")>=0) showAbout=false; }
        EndDrawing();
    }
    if(scene.loaded) UnloadModel(scene.model);
    CloseWindow();
    return 0;
}
