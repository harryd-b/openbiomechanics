/**
 * Three.js Skeleton Viewer for Biomechanics
 * Renders H36M skeleton with smooth animation
 */

// H36M skeleton connections
const SKELETON_CONNECTIONS = [
    [0, 7], [7, 8], [8, 9], [9, 10],  // Spine
    [8, 11], [11, 12], [12, 13],      // Left arm
    [8, 14], [14, 15], [15, 16],      // Right arm
    [0, 4], [4, 5], [5, 6],           // Left leg
    [0, 1], [1, 2], [2, 3],           // Right leg
];

class SkeletonViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) return;

        this.poses = [];
        this.currentFrame = 0;
        this.isPlaying = false;
        this.fps = 30;

        this.init();
        this.animate();
    }

    init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1e1e32);

        // Camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 100);
        this.camera.position.set(2, 1.5, 2);
        this.camera.lookAt(0, 0.8, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // Lights
        const ambient = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambient);

        const directional = new THREE.DirectionalLight(0xffffff, 1);
        directional.position.set(5, 10, 5);
        directional.castShadow = true;
        directional.shadow.mapSize.width = 2048;
        directional.shadow.mapSize.height = 2048;
        this.scene.add(directional);

        // Floor grid
        this.createFloor();

        // Skeleton group
        this.skeletonGroup = new THREE.Group();
        this.scene.add(this.skeletonGroup);

        // Shadow plane
        const shadowGeo = new THREE.PlaneGeometry(10, 10);
        const shadowMat = new THREE.ShadowMaterial({ opacity: 0.3 });
        this.shadowPlane = new THREE.Mesh(shadowGeo, shadowMat);
        this.shadowPlane.rotation.x = -Math.PI / 2;
        this.shadowPlane.position.y = 0.001;
        this.shadowPlane.receiveShadow = true;
        this.scene.add(this.shadowPlane);

        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.target.set(0, 0.8, 0);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.update();

        // Handle resize
        window.addEventListener('resize', () => this.onResize());

        // Create skeleton meshes
        this.createSkeleton();
    }

    createFloor() {
        // Grid helper with fading
        const gridSize = 4;
        const gridDivisions = 20;

        const gridHelper = new THREE.GridHelper(gridSize, gridDivisions, 0x444466, 0x333344);
        gridHelper.position.y = 0;
        this.scene.add(gridHelper);

        // Fade grid at edges
        const gridMaterial = gridHelper.material;
        if (Array.isArray(gridMaterial)) {
            gridMaterial.forEach(m => {
                m.transparent = true;
                m.opacity = 0.5;
            });
        }
    }

    createSkeleton() {
        this.joints = [];
        this.bones = [];

        // Joint spheres
        const jointGeo = new THREE.SphereGeometry(0.025, 16, 16);
        const jointMat = new THREE.MeshStandardMaterial({
            color: 0xff6666,
            emissive: 0x331111,
            roughness: 0.5
        });

        for (let i = 0; i < 17; i++) {
            const joint = new THREE.Mesh(jointGeo, jointMat);
            joint.castShadow = true;
            this.joints.push(joint);
            this.skeletonGroup.add(joint);
        }

        // Bone cylinders
        const boneMat = new THREE.MeshStandardMaterial({
            color: 0xff4444,
            emissive: 0x220000,
            roughness: 0.4
        });

        for (const [start, end] of SKELETON_CONNECTIONS) {
            const boneGeo = new THREE.CylinderGeometry(0.012, 0.012, 1, 8);
            const bone = new THREE.Mesh(boneGeo, boneMat);
            bone.castShadow = true;
            this.bones.push({ mesh: bone, start, end });
            this.skeletonGroup.add(bone);
        }
    }

    updateSkeleton(pose) {
        if (!pose || pose.length < 17) return;

        // Update joint positions
        for (let i = 0; i < Math.min(17, pose.length); i++) {
            const [x, y, z] = pose[i];
            this.joints[i].position.set(x, z, -y); // Swap Y/Z for Three.js
        }

        // Update bone positions and orientations
        for (const bone of this.bones) {
            const startPos = this.joints[bone.start].position;
            const endPos = this.joints[bone.end].position;

            // Position at midpoint
            bone.mesh.position.copy(startPos).add(endPos).multiplyScalar(0.5);

            // Orient towards end
            const direction = new THREE.Vector3().subVectors(endPos, startPos);
            const length = direction.length();

            bone.mesh.scale.y = length;

            // Align cylinder to direction
            const axis = new THREE.Vector3(0, 1, 0);
            const quaternion = new THREE.Quaternion();
            quaternion.setFromUnitVectors(axis, direction.normalize());
            bone.mesh.quaternion.copy(quaternion);
        }
    }

    setPoses(poses) {
        this.poses = poses;
        this.currentFrame = 0;
        if (poses.length > 0) {
            this.updateSkeleton(poses[0]);
        }
    }

    setFrame(frameIndex) {
        if (frameIndex >= 0 && frameIndex < this.poses.length) {
            this.currentFrame = frameIndex;
            this.updateSkeleton(this.poses[frameIndex]);
        }
    }

    play() {
        this.isPlaying = true;
    }

    pause() {
        this.isPlaying = false;
    }

    onResize() {
        if (!this.container) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        if (this.isPlaying && this.poses.length > 0) {
            this.currentFrame = (this.currentFrame + 1) % this.poses.length;
            this.updateSkeleton(this.poses[this.currentFrame]);
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

// Global instance
let skeletonViewer = null;

// Initialize when DOM is ready
function initSkeletonViewer(containerId) {
    skeletonViewer = new SkeletonViewer(containerId);
    return skeletonViewer;
}

// Update poses from Dash callback
function updateSkeletonPoses(posesJson) {
    if (skeletonViewer && posesJson) {
        try {
            const poses = JSON.parse(posesJson);
            skeletonViewer.setPoses(poses);
        } catch (e) {
            console.error('Failed to parse poses:', e);
        }
    }
}

// Set current frame from Dash
function setSkeletonFrame(frameIndex) {
    if (skeletonViewer) {
        skeletonViewer.setFrame(frameIndex);
    }
}

// Export for use
window.SkeletonViewer = SkeletonViewer;
window.initSkeletonViewer = initSkeletonViewer;
window.updateSkeletonPoses = updateSkeletonPoses;
window.setSkeletonFrame = setSkeletonFrame;
