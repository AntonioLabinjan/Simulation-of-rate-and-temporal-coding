# Simulacija: Event-based input -> enkodiranje spikeova -> Spiking convolutional layer (LIF neurons)
# Ovaj kod radi dvije stvari:
# 1) Ako je `USE_WEBCAM = False` onda generira sintetičku sekvencu (pomaknutu vertikalnu crtu) i vrti kratku demo animaciju.
# 2) Ako je `USE_WEBCAM = True` pokušat će otvoriti web kameru i real-time procesirati frameove u event-map -> spike -> spiking-conv.
#
# Ovisnosti: numpy, matplotlib, opencv-python
# Pokreni lokalno: pip install numpy matplotlib opencv-python
#
# Napomena: okruženje u kojem se izvršava ovaj kod možda nema pristup web kameri — zato defaultno koristim sintetički primjer.
# Ako želiš pokrenuti stvarni webcam, postavi USE_WEBCAM = True.

# OVO SIMULIRA RATE CODING => 1 EVENT = 1 SPIKE
#Rate coding: čim se dogodi event → spike odmah.

#Temporal coding: jak event → spike odmah, slab event → spike s delayem.

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation

# ---------- PARAMETERS ----------
USE_WEBCAM = False  # promijeni na True za lokalnu kameru (ako je dostupna)
FRAME_W, FRAME_H = 64, 64
EVENT_THRESHOLD = 20       # prag razlike za event detekciju (intensity delta)
SPIKE_TIME_WINDOW = 20     # broj vremenskih koraka za koje mapiramo spikeove
LIF_TAU = 5.0              # konstanta curenja za LIF
LIF_THRESHOLD = 1.0        # threshold za ispaljivanje
LIF_RESET = 0.0            # reset value nakon spike
CONV_KERNEL = np.array([[0.0, -1.0, 0.0],
                        [-1.0, 4.0, -1.0],
                        [0.0, -1.0, 0.0]])  # koristimo laplacian-like kernel kao težine (primjer)

# ---------- HELPERS ----------
def frame_to_gray(frame, size=(FRAME_W, FRAME_H)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return gray.astype(np.int16)

def event_from_frames(prev, cur, threshold=EVENT_THRESHOLD):
    # jednostavan event-detektor: ako se intenzitet promijeni više od threshold => event
    diff = cur - prev
    pos = (diff >= threshold).astype(np.int8)
    neg = (diff <= -threshold).astype(np.int8)
    # event map: +1 for ON, -1 for OFF, 0 for none
    events = pos - neg
    return events, diff

def events_to_spiketrains(events_sequence, window=SPIKE_TIME_WINDOW):
    # events_sequence: list of event maps over time (shape T x H x W)
    # vratimo spike tensor: (T x H x W) binarna (1=spike)
    T = len(events_sequence)
    H, W = events_sequence[0].shape
    spikes = np.zeros((T, H, W), dtype=np.uint8)
    # Simple temporal coding: event at time t -> single spike at t (or could be mapped to multiple times)
    for t, ev in enumerate(events_sequence):
        spikes[t] = (ev != 0).astype(np.uint8)
    return spikes

def conv2d_spike_map(spike_map, kernel):
    # konvolucija jednog 2D spike_map-a s kernelom, 'same' padding
    return cv2.filter2D(spike_map.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE)

# ---------- LIF Spiking "Convolutional" Layer ----------
class SpikingConvLayer:
    def __init__(self, kernel, tau=LIF_TAU, threshold=LIF_THRESHOLD):
        self.kernel = kernel.astype(np.float32)
        self.tau = tau
        self.threshold = threshold
        self.potential = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        self.refractory = np.zeros_like(self.potential)
    
    def step(self, input_spike_map):
        # input_spike_map: 2D binary map
        # 1) konvoluiraj spike map s kernelom -> postsynaptic currents
        current = conv2d_spike_map(input_spike_map, self.kernel)
        # 2) update LIF potential (discrete leaky integrator)
        # dp/dt = -p/tau + current
        self.potential += (-self.potential / self.tau) + current
        # 3) spike generation
        spikes = (self.potential >= self.threshold).astype(np.uint8)
        # reset potentials where spiked
        self.potential[spikes.astype(bool)] = LIF_RESET
        return spikes, self.potential.copy()

# ---------- Synthetic sequence generator ----------
def synth_sequence(num_frames=60, w=FRAME_W, h=FRAME_H):
    seq = []
    # moving vertical bar
    bar_w = 3
    for t in range(num_frames):
        frame = np.ones((h, w), dtype=np.uint8) * 10
        x = (t * 2) % (w + bar_w) - bar_w
        xs = max(0, x)
        xe = min(w, x + bar_w)
        if xs < xe:
            frame[:, xs:xe] = 200
        seq.append(frame)
    return seq

# ---------- Demo runner (sinteticka bez kamere) ----------
def run_demo_synthetic():
    seq = synth_sequence(80)
    prev = frame_to_gray(seq[0])
    events_seq = []
    # build event maps from sequence
    for cur_frame in seq[1:]:
        cur = frame_to_gray(cur_frame)
        ev_map, diff = event_from_frames(prev, cur)
        events_seq.append(ev_map)
        prev = cur
    # encode spikes (simple temporal coding)
    spikes_tensor = events_to_spiketrains(events_seq)
    # create spiking conv layer
    layer = SpikingConvLayer(CONV_KERNEL)
    # simulate through time and collect visualizations
    fired_maps = []
    potentials = []
    for t in range(len(spikes_tensor)):
        spikes_t = spikes_tensor[t]
        fired, pot = layer.step(spikes_t)
        fired_maps.append(fired)
        potentials.append(pot)
    # plot sample frames: original, events sum, fired sum
    fig, axes = plt.subplots(1,4, figsize=(12,3))
    axes[0].set_title("Example frame (center)")
    axes[0].imshow(seq[len(seq)//2], cmap='gray', vmin=0, vmax=255)
    axes[0].axis('off')
    axes[1].set_title("Event count (sum over time)")
    axes[1].imshow(np.sum(np.abs(spikes_tensor), axis=0), cmap='hot')
    axes[1].axis('off')
    axes[2].set_title("Spikes fired (sum)")
    axes[2].imshow(np.sum(fired_maps, axis=0), cmap='hot')
    axes[2].axis('off')
    axes[3].set_title("Final membrane potential")
    axes[3].imshow(potentials[-1], cmap='viridis')
    axes[3].axis('off')
    plt.tight_layout()
    plt.show()

# ---------- Webcam runner (if available) ----------
def run_demo_webcam(max_steps=500):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ne mogu otvoriti kameru. Pokrećem sintetički demo umjesto toga.")
        run_demo_synthetic()
        return
    ret, frame = cap.read()
    if not ret:
        print("Ne mogu pročitati frame iz kamere. Prekid.")
        cap.release()
        return
    prev = frame_to_gray(frame)
    layer = SpikingConvLayer(CONV_KERNEL)
    step = 0
    try:
        while step < max_steps:
            ret, frame = cap.read()
            if not ret:
                break
            cur = frame_to_gray(frame)
            events_map, diff = event_from_frames(prev, cur)
            spikes_map = (events_map != 0).astype(np.uint8)
            fired, pot = layer.step(spikes_map)
            # vizualizacija s OpenCV
            vis = np.zeros((FRAME_H, FRAME_W*3), dtype=np.uint8)
            vis[:, :FRAME_W] = cur.astype(np.uint8)
            vis[:, FRAME_W:FRAME_W*2] = (np.abs(diff).clip(0,255)).astype(np.uint8)
            vis[:, FRAME_W*2:] = (fired*255).astype(np.uint8)
            vis_up = cv2.resize(vis, (FRAME_W*6, FRAME_H*6), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('gray | diff | fired', vis_up)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            prev = cur.copy()
            step += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    if USE_WEBCAM:
        run_demo_webcam()
    else:
        run_demo_synthetic()
