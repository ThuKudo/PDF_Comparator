import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  FileText, Play, Save, Download, LayoutTemplate, Settings, 
  ZoomIn, ZoomOut, CheckCircle, EyeOff,
  ChevronLeft, ChevronRight, Filter, Layers, List,
  Upload, X, Maximize2, Minimize2, Eye
} from 'lucide-react';
import * as pdfjsLib from 'pdfjs-dist';
import pixelmatch from 'pixelmatch';

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url
).toString();

// --- Render PDF page to canvas ---
async function renderPdfPageToCanvas(pdfDoc, pageNum, scale = 1.5) {
  const page = await pdfDoc.getPage(pageNum);
  const viewport = page.getViewport({ scale });
  const canvas = document.createElement('canvas');
  canvas.width = viewport.width;
  canvas.height = viewport.height;
  const ctx = canvas.getContext('2d');
  await page.render({ canvasContext: ctx, viewport }).promise;
  return canvas;
}

// --- Crop a region from a canvas and return a dataURL ---
function cropCanvas(canvas, box, padding = 30) {
  const w = canvas.width;
  const h = canvas.height;
  const x1 = Math.max(0, Math.floor((box.x / 100) * w) - padding);
  const y1 = Math.max(0, Math.floor((box.y / 100) * h) - padding);
  const x2 = Math.min(w, Math.ceil(((box.x + box.w) / 100) * w) + padding);
  const y2 = Math.min(h, Math.ceil(((box.y + box.h) / 100) * h) + padding);
  const cw = x2 - x1;
  const ch = y2 - y1;
  if (cw <= 0 || ch <= 0) return null;
  const crop = document.createElement('canvas');
  crop.width = cw;
  crop.height = ch;
  crop.getContext('2d').drawImage(canvas, x1, y1, cw, ch, 0, 0, cw, ch);
  return crop.toDataURL();
}

// --- Cluster diff pixels into bounding boxes ---
function clusterDiffPixels(diffCanvas, minClusterPixels = 20) {
  const w = diffCanvas.width;
  const h = diffCanvas.height;
  const ctx = diffCanvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;

  const cellSize = 4;
  const cols = Math.ceil(w / cellSize);
  const rows = Math.ceil(h / cellSize);
  const grid = new Uint8Array(cols * rows);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      const r = data[idx], g = data[idx+1], b = data[idx+2], a = data[idx+3];
      if (a > 0 && (r > 100 || g > 100) && !(r > 200 && g > 200 && b > 200)) {
        const cx = Math.floor(x / cellSize);
        const cy = Math.floor(y / cellSize);
        grid[cy * cols + cx] = 1;
      }
    }
  }

  // Morphological dilation: expand each active cell outward by 'radius' 
  // to merge nearby small clusters into larger meaningful ones
  const dilateRadius = 5;
  const dilated = new Uint8Array(cols * rows);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      if (grid[y * cols + x] === 1) {
        for (let dy = -dilateRadius; dy <= dilateRadius; dy++) {
          for (let dx = -dilateRadius; dx <= dilateRadius; dx++) {
            const nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
              dilated[ny * cols + nx] = 1;
            }
          }
        }
      }
    }
  }

  // Flood-fill on dilated grid
  const visited = new Uint8Array(cols * rows);
  const clusters = [];

  for (let i = 0; i < dilated.length; i++) {
    if (dilated[i] === 1 && !visited[i]) {
      const queue = [i];
      visited[i] = 1;
      let minX = cols, minY = rows, maxX = 0, maxY = 0;
      let realPixelCount = 0;

      const MAX_W = cols / 3;
      const MAX_H = rows / 3;

      while (queue.length > 0) {
        const cur = queue.shift();
        const cx = cur % cols;
        const cy = Math.floor(cur / cols);
        minX = Math.min(minX, cx);
        minY = Math.min(minY, cy);
        maxX = Math.max(maxX, cx);
        maxY = Math.max(maxY, cy);
        // Count only REAL diff pixels (not dilated ones)
        if (grid[cur] === 1) realPixelCount++;

        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = cx + dx, ny = cy + dy;
            if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
              const ni = ny * cols + nx;
              if (dilated[ni] === 1 && !visited[ni]) {
                const potentialMinX = Math.min(minX, nx);
                const potentialMaxX = Math.max(maxX, nx);
                const potentialMinY = Math.min(minY, ny);
                const potentialMaxY = Math.max(maxY, ny);
                
                // Chunk the diff cluster if it exceeds maximum allowed bounding box size
                if ((potentialMaxX - potentialMinX <= MAX_W) && (potentialMaxY - potentialMinY <= MAX_H)) {
                  visited[ni] = 1;
                  queue.push(ni);
                }
              }
            }
          }
        }
      }

      const pad = 15;
      const boxX = Math.max(0, minX * cellSize - pad);
      const boxY = Math.max(0, minY * cellSize - pad);
      const boxW = Math.min(w, (maxX + 1) * cellSize + pad) - boxX;
      const boxH = Math.min(h, (maxY + 1) * cellSize + pad) - boxY;

      if (realPixelCount >= minClusterPixels) {
        clusters.push({
          x: (boxX / w) * 100,
          y: (boxY / h) * 100,
          w: (boxW / w) * 100,
          h: (boxH / h) * 100,
          pixelCount: realPixelCount,
          area: boxW * boxH,
        });
      }
    }
  }

  // Sort by area descending (most important first)
  clusters.sort((a, b) => b.area - a.area);
  return clusters;
}

function getSeverity(cluster) {
  if (cluster.area > 20000) return 'High';
  if (cluster.area > 5000) return 'Medium';
  return 'Low';
}

// --- Extract bounding box of non-white content ---
function getContentBoundingBox(canvas) {
  const width = canvas.width;
  const height = canvas.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const data = ctx.getImageData(0, 0, width, height).data;
  
  let minX = width, minY = height, maxX = 0, maxY = 0;
  
  // Downsample scan for performance (step by 4 pixels)
  for (let y = 0; y < height; y += 4) {
    for (let x = 0; x < width; x += 4) {
      const idx = (y * width + x) * 4;
      // Not purely white
      if (data[idx] < 250 || data[idx+1] < 250 || data[idx+2] < 250) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  
  if (minX > maxX || minY > maxY) return { x: 0, y: 0, w: width, h: height };
  
  return {
    x: Math.max(0, minX - 4),
    y: Math.max(0, minY - 4),
    w: Math.min(width, maxX - minX + 8),
    h: Math.min(height, maxY - minY + 8)
  };
}

// --- Compute similarity between two canvases (0=identical, 1=completely different) ---
function computeSimilarity(canvasA, canvasB) {
  const w = Math.max(canvasA.width, canvasB.width);
  const h = Math.max(canvasA.height, canvasB.height);
  const normA = document.createElement('canvas'); normA.width = w; normA.height = h;
  const ctxA = normA.getContext('2d'); ctxA.fillStyle = 'white'; ctxA.fillRect(0,0,w,h); ctxA.drawImage(canvasA,0,0);
  const normB = document.createElement('canvas'); normB.width = w; normB.height = h;
  const ctxB = normB.getContext('2d'); ctxB.fillStyle = 'white'; ctxB.fillRect(0,0,w,h); ctxB.drawImage(canvasB,0,0);
  const diff = new Uint8ClampedArray(w * h * 4);
  const numDiff = pixelmatch(ctxA.getImageData(0,0,w,h).data, ctxB.getImageData(0,0,w,h).data, diff, w, h, { threshold: 0.3 });
  return numDiff / (w * h); // 0 = same, 1 = all different
}

// --- Match pages between two PDFs using content similarity ---
async function matchPages(pdfA, pdfB, progressCb) {
  const thumbScale = 0.3; // Small thumbnails for fast comparison
  const pagesA = pdfA.numPages;
  const pagesB = pdfB.numPages;
  
  progressCb('Rendering thumbnails File A...');
  const thumbsA = [];
  for (let i = 1; i <= pagesA; i++) {
    thumbsA.push(await renderPdfPageToCanvas(pdfA, i, thumbScale));
  }
  
  progressCb('Rendering thumbnails File B...');
  const thumbsB = [];
  for (let i = 1; i <= pagesB; i++) {
    thumbsB.push(await renderPdfPageToCanvas(pdfB, i, thumbScale));
  }
  
  progressCb('Matching pages...');
  // Compute similarity matrix
  const matrix = [];
  for (let a = 0; a < pagesA; a++) {
    matrix[a] = [];
    for (let b = 0; b < pagesB; b++) {
      matrix[a][b] = computeSimilarity(thumbsA[a], thumbsB[b]);
    }
  }
  
  // Greedy matching: for each page A, find best matching page B
  // Constraint: preserve order (page mapping should be monotonically increasing)
  const MATCH_THRESHOLD = 0.15; // Max 15% pixels different to be considered a match
  const matchedB = new Set();
  const pageMap = []; // Array of { pageA, pageB, similarity, type }
  
  // Forward pass: match A pages to B pages in order
  let lastMatchedB = -1;
  for (let a = 0; a < pagesA; a++) {
    let bestB = -1;
    let bestSim = Infinity;
    for (let b = lastMatchedB + 1; b < pagesB; b++) {
      if (!matchedB.has(b) && matrix[a][b] < bestSim) {
        bestSim = matrix[a][b];
        bestB = b;
      }
    }
    
    if (bestB >= 0 && bestSim < MATCH_THRESHOLD) {
      // Check if any unmatched B pages before bestB are 'new pages in B'
      for (let b = lastMatchedB + 1; b < bestB; b++) {
        if (!matchedB.has(b)) {
          pageMap.push({ pageA: null, pageB: b + 1, similarity: 1, type: 'added' });
          matchedB.add(b);
        }
      }
      pageMap.push({ pageA: a + 1, pageB: bestB + 1, similarity: bestSim, type: bestSim < 0.001 ? 'identical' : 'modified' });
      matchedB.add(bestB);
      lastMatchedB = bestB;
    } else {
      pageMap.push({ pageA: a + 1, pageB: null, similarity: 1, type: 'removed' });
    }
  }
  
  // Any remaining unmatched B pages
  for (let b = 0; b < pagesB; b++) {
    if (!matchedB.has(b)) {
      pageMap.push({ pageA: null, pageB: b + 1, similarity: 1, type: 'added' });
    }
  }
  
  // Sort by visual order (by the first non-null page number)
  pageMap.sort((a, b) => {
    const orderA = a.pageA || a.pageB;
    const orderB = b.pageA || b.pageB;
    return orderA - orderB;
  });
  
  return pageMap;
}


export default function PdfComparatorApp() {
  const [isComparing, setIsComparing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState('');
  const [diffs, setDiffs] = useState([]);
  const [selectedDiffId, setSelectedDiffId] = useState(null);
  const [zoom, setZoom] = useState(100);
  const [filterStatus, setFilterStatus] = useState('All');
  const [focusMode, setFocusMode] = useState(true);
  const [cropZoom, setCropZoom] = useState(100);
  const [sidebarWidth, setSidebarWidth] = useState(300);
  const [hideAccepted, setHideAccepted] = useState(true);
  const [detailPanelHeight, setDetailPanelHeight] = useState(280);
  const [pageMap, setPageMap] = useState([]); // Smart page matching results
  const [isSetupOpen, setIsSetupOpen] = useState(false);
  const [proposedMapping, setProposedMapping] = useState([]);
  const isResizing = useRef(false);

  // PDF
  const [fileA, setFileA] = useState(null);
  const [fileB, setFileB] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const renderScale = 2.0; // Higher for better quality crops

  // Store raw canvases for cropping
  const canvasCacheA = useRef({});
  const canvasCacheB = useRef({});

  // Drag-drop
  const [dragOverA, setDragOverA] = useState(false);
  const [dragOverB, setDragOverB] = useState(false);
  const fileInputARef = useRef(null);
  const fileInputBRef = useRef(null);

  // Sync scroll
  const viewerARef = useRef(null);
  const viewerBRef = useRef(null);
  const isSyncingLeft = useRef(false);
  const isSyncingRight = useRef(false);

  // --- Load PDF ---
  const loadPdf = useCallback(async (file, setter, cacheRef) => {
    if (!file || file.type !== 'application/pdf') {
      alert('Vui lòng chọn file PDF hợp lệ.');
      return;
    }
    const arrayBuffer = await file.arrayBuffer();
    const pdfDoc = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    const totalPages = pdfDoc.numPages;
    const canvas = await renderPdfPageToCanvas(pdfDoc, 1, renderScale);
    cacheRef.current = { 1: canvas };
    setter({
      name: file.name,
      pdfDoc,
      pages: { 1: canvas.toDataURL() },
      totalPages,
    });
  }, [renderScale]);

  const ensurePageRendered = useCallback(async (fileState, setter, pageNum, cacheRef) => {
    if (!fileState || !fileState.pdfDoc || fileState.pages[pageNum]) return;
    const canvas = await renderPdfPageToCanvas(fileState.pdfDoc, pageNum, renderScale);
    cacheRef.current[pageNum] = canvas;
    setter(prev => ({ ...prev, pages: { ...prev.pages, [pageNum]: canvas.toDataURL() } }));
  }, [renderScale]);

  useEffect(() => {
    if (fileA) ensurePageRendered(fileA, setFileA, currentPage, canvasCacheA);
    if (fileB) ensurePageRendered(fileB, setFileB, currentPage, canvasCacheB);
  }, [currentPage, fileA?.pdfDoc, fileB?.pdfDoc]);

  const handleFileSelect = (e, setter, cacheRef) => {
    const file = e.target.files[0];
    if (file) loadPdf(file, setter, cacheRef);
  };

  const handleDragOver = (e) => { e.preventDefault(); e.stopPropagation(); };
  const handleDropA = (e) => { e.preventDefault(); e.stopPropagation(); setDragOverA(false); const f = e.dataTransfer.files[0]; if (f) loadPdf(f, setFileA, canvasCacheA); };
  const handleDropB = (e) => { e.preventDefault(); e.stopPropagation(); setDragOverB(false); const f = e.dataTransfer.files[0]; if (f) loadPdf(f, setFileB, canvasCacheB); };

  // ============================================================
  // COMPARE ENGINE with Smart Page Matching
  // ============================================================
  const openMappingSetup = async () => {
    if (!fileA || !fileB) return;

    setIsComparing(true);
    setProgress(0);
    setProgressText('Đang phân tích cấu trúc trang...');
    try {
      const mapping = await matchPages(fileA.pdfDoc, fileB.pdfDoc, setProgressText);
      setProposedMapping(mapping);
      setIsSetupOpen(true);
    } catch (e) {
      alert("Lỗi phân tích: " + e.message);
    } finally {
      setIsComparing(false);
    }
  };

  const executeCompare = async () => {
    setIsSetupOpen(false);
    setIsComparing(true);
    setProgress(0);
    setProgressText('Đang chuẩn bị so sánh...');
    setDiffs([]);
    setSelectedDiffId(null);
    setPageMap(proposedMapping);

    const allDiffs = [];
    let diffCounter = 0;

    try {
      setProgress(20);

      const matchedPairs = proposedMapping.filter(m => m.type === 'modified' || m.type === 'identical');
      const addedPages = proposedMapping.filter(m => m.type === 'added');
      const removedPages = proposedMapping.filter(m => m.type === 'removed');

      // --- Phase 2: Compare matched page pairs ---
      const totalWork = matchedPairs.length + addedPages.length + removedPages.length;
      let workDone = 0;

      for (const pair of matchedPairs) {
        if (pair.type === 'identical') {
          workDone++;
          setProgress(20 + Math.round((workDone / totalWork) * 75));
          continue; // Skip identical pages
        }

        setProgressText(`So sánh trang A:${pair.pageA} ↔ B:${pair.pageB}...`);

        const canvasA = await renderPdfPageToCanvas(fileA.pdfDoc, pair.pageA, renderScale);
        const canvasB = await renderPdfPageToCanvas(fileB.pdfDoc, pair.pageB, renderScale);
        canvasCacheA.current[pair.pageA] = canvasA;
        canvasCacheB.current[pair.pageB] = canvasB;

        setFileA(prev => ({ ...prev, pages: { ...prev.pages, [pair.pageA]: canvasA.toDataURL() } }));
        setFileB(prev => ({ ...prev, pages: { ...prev.pages, [pair.pageB]: canvasB.toDataURL() } }));

        // ----------------------------------------------------
        // SMART ALIGNMENT: Content Bounding Box Normalization
        // ----------------------------------------------------
        setProgressText(`Chuẩn hóa khung bản vẽ trang A:${pair.pageA}...`);
        const boxA = getContentBoundingBox(canvasA);
        const boxB = getContentBoundingBox(canvasB);

        // Compute dimension deviation
        const scaleX = boxA.w / Math.max(1, boxB.w);
        const scaleY = boxA.h / Math.max(1, boxB.h);
        const maxDeviation = Math.max(Math.abs(1 - scaleX), Math.abs(1 - scaleY));

        const w = Math.max(canvasA.width, canvasB.width);
        const h = Math.max(canvasA.height, canvasB.height);

        const normA = document.createElement('canvas'); normA.width = w; normA.height = h;
        const ctxA = normA.getContext('2d'); ctxA.fillStyle = 'white'; ctxA.fillRect(0,0,w,h); ctxA.drawImage(canvasA,0,0);

        const normB = document.createElement('canvas'); normB.width = w; normB.height = h;
        const ctxB = normB.getContext('2d'); ctxB.fillStyle = 'white'; ctxB.fillRect(0,0,w,h);
        
        // If deviation > 5%, it's likely a completely different layout, skip auto-align
        if (maxDeviation <= 0.05) {
          // Force Canvas B's content to exactly overlap Canvas A's content bounding box
          ctxB.drawImage(canvasB, boxB.x, boxB.y, boxB.w, boxB.h, boxA.x, boxA.y, boxA.w, boxA.h);
        } else {
          ctxB.drawImage(canvasB, 0, 0);
        }

        const diffCanvas = document.createElement('canvas'); diffCanvas.width = w; diffCanvas.height = h;
        const diffCtx = diffCanvas.getContext('2d');
        const diffImageData = diffCtx.createImageData(w, h);

        const numDiffPixels = pixelmatch(
          ctxA.getImageData(0,0,w,h).data, ctxB.getImageData(0,0,w,h).data,
          diffImageData.data, w, h,
          { threshold: 0.2, alpha: 0.3, diffColor: [255, 50, 50] }
        );
        diffCtx.putImageData(diffImageData, 0, 0);

        if (numDiffPixels > 100) {
          const clusters = clusterDiffPixels(diffCanvas, 30);
          for (const cluster of clusters) {
            diffCounter++;
            const box = { x: cluster.x, y: cluster.y, w: cluster.w, h: cluster.h };
            allDiffs.push({
              id: `diff-${String(diffCounter).padStart(3, '0')}`,
              type: 'Visual', severity: getSeverity(cluster), status: 'New',
              pageA: pair.pageA, pageB: pair.pageB, box,
              confidence: Math.min(0.99, 0.5 + (cluster.pixelCount / 300)),
              description: `Thay đổi #${diffCounter} (A:${pair.pageA} ↔ B:${pair.pageB})`,
              pixelCount: cluster.pixelCount,
              cropA: cropCanvas(canvasA, box, 80),
              cropB: cropCanvas(canvasB, box, 80),
              reviewerNote: '',
            });
          }
        }
        workDone++;
        setProgress(20 + Math.round((workDone / totalWork) * 75));
      }

      // --- Phase 3: Flag added pages (only in File B) ---
      for (const added of addedPages) {
        diffCounter++;
        const canvasB = await renderPdfPageToCanvas(fileB.pdfDoc, added.pageB, renderScale);
        canvasCacheB.current[added.pageB] = canvasB;
        setFileB(prev => ({ ...prev, pages: { ...prev.pages, [added.pageB]: canvasB.toDataURL() } }));
        allDiffs.push({
          id: `diff-${String(diffCounter).padStart(3, '0')}`,
          type: 'PageAdded', severity: 'High', status: 'New',
          pageA: null, pageB: added.pageB,
          box: { x: 0, y: 0, w: 100, h: 100 },
          confidence: 1.0,
          description: `📄 Trang MỚI #${added.pageB} (chỉ có trong File B)`,
          pixelCount: 0, cropA: null, cropB: canvasB.toDataURL(), reviewerNote: '',
        });
        workDone++;
        setProgress(20 + Math.round((workDone / totalWork) * 75));
      }

      // --- Phase 4: Flag removed pages (only in File A) ---
      for (const removed of removedPages) {
        diffCounter++;
        const canvasA = await renderPdfPageToCanvas(fileA.pdfDoc, removed.pageA, renderScale);
        canvasCacheA.current[removed.pageA] = canvasA;
        setFileA(prev => ({ ...prev, pages: { ...prev.pages, [removed.pageA]: canvasA.toDataURL() } }));
        allDiffs.push({
          id: `diff-${String(diffCounter).padStart(3, '0')}`,
          type: 'PageRemoved', severity: 'High', status: 'New',
          pageA: removed.pageA, pageB: null,
          box: { x: 0, y: 0, w: 100, h: 100 },
          confidence: 1.0,
          description: `🗑️ Trang #${removed.pageA} đã BỊ XÓA (chỉ có trong File A)`,
          pixelCount: 0, cropA: canvasA.toDataURL(), cropB: null, reviewerNote: '',
        });
        workDone++;
        setProgress(20 + Math.round((workDone / totalWork) * 75));
      }

      setDiffs(allDiffs);
      const identicalCount = matchedPairs.filter(m => m.type === 'identical').length;
      setProgressText(`Hoàn tất! ${allDiffs.length} thay đổi · ${identicalCount} trang giống hệt · ${addedPages.length} trang mới · ${removedPages.length} trang bị xóa`);
      setProgress(100);
      if (allDiffs.length > 0) setSelectedDiffId(allDiffs[0].id);

    } catch (err) {
      console.error(err);
      setProgressText(`Lỗi: ${err.message}`);
    } finally {
      setTimeout(() => setIsComparing(false), 400);
    }
  };

  const handleStatusChange = (id, newStatus) => {
    // Find next issue before updating state
    const currentIdx = filteredDiffs.findIndex(d => d.id === id);
    const remaining = filteredDiffs.filter(d => d.id !== id && d.status === 'New');
    
    setDiffs(prev => prev.map(d => d.id === id ? { ...d, status: newStatus } : d));
    
    // Auto-navigate to next unprocessed issue
    if (remaining.length > 0) {
      // Pick the next one in list, or loop back to first
      const nextInList = filteredDiffs.find((d, i) => i > currentIdx && d.id !== id && d.status === 'New')
        || remaining[0];
      if (nextInList) {
        setTimeout(() => handleSelectDiff(nextInList.id), 50);
      }
    } else {
      setSelectedDiffId(null);
    }
  };

  const filteredDiffs = diffs.filter(d => {
    if (filterStatus !== 'All' && d.status !== filterStatus) return false;
    if (hideAccepted && d.status === 'Accepted') return false;
    return true;
  });
  const selectedDiff = diffs.find(d => d.id === selectedDiffId);

  const maxPages = Math.max(fileA?.totalPages || 1, fileB?.totalPages || 1);
  const goToPrevPage = () => setCurrentPage(p => Math.max(1, p - 1));
  const goToNextPage = () => setCurrentPage(p => Math.min(maxPages, p + 1));

  // --- Click issue -> zoom to that area ---
  const handleSelectDiff = (diffId) => {
    setSelectedDiffId(diffId);
    const diff = diffs.find(d => d.id === diffId);
    if (diff) {
      setCurrentPage(diff.pageA || diff.pageB || 1);
      // Scroll viewers to center on the diff box
      setTimeout(() => {
        [viewerARef, viewerBRef].forEach(ref => {
          const el = ref.current;
          if (!el) return;
          const img = el.querySelector('img');
          if (!img) return;
          const scale = zoom / 100;
          const imgW = img.naturalWidth * scale;
          const imgH = img.naturalHeight * scale;
          const targetX = (diff.box.x / 100) * imgW - el.clientWidth / 2 + (diff.box.w / 100) * imgW / 2;
          const targetY = (diff.box.y / 100) * imgH - el.clientHeight / 2 + (diff.box.h / 100) * imgH / 2;
          el.scrollTo({ left: Math.max(0, targetX), top: Math.max(0, targetY), behavior: 'smooth' });
        });
      }, 100);
    }
  };

  const navigateDiff = (direction) => {
    if (filteredDiffs.length === 0) return;
    const currentIdx = filteredDiffs.findIndex(d => d.id === selectedDiffId);
    let nextIdx;
    if (direction === 'next') nextIdx = currentIdx < filteredDiffs.length - 1 ? currentIdx + 1 : 0;
    else nextIdx = currentIdx > 0 ? currentIdx - 1 : filteredDiffs.length - 1;
    handleSelectDiff(filteredDiffs[nextIdx].id);
  };

  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === 'n' || e.key === 'N') { e.preventDefault(); navigateDiff('next'); }
      if (e.key === 'p' || e.key === 'P') { e.preventDefault(); navigateDiff('prev'); }
      if ((e.key === 'a' || e.key === 'A') && selectedDiffId) handleStatusChange(selectedDiffId, 'Accepted');
      if ((e.key === 'i' || e.key === 'I') && selectedDiffId) handleStatusChange(selectedDiffId, 'Ignored');
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [selectedDiffId, filteredDiffs, diffs]);

  // Global Zoom
  useEffect(() => {
    const handleWheelZoom = (e) => {
      const inViewer = e.target.closest('.js-viewer-container');
      const inDetail = e.target.closest('.js-detail-panel');
      
      if (inViewer && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        setZoom(z => Math.max(20, Math.min(500, e.deltaY < 0 ? z + 15 : z - 15)));
      } else if (inDetail) {
        e.preventDefault();
        setCropZoom(z => Math.max(50, Math.min(800, e.deltaY < 0 ? z + 15 : z - 15)));
      }
    };
    document.addEventListener('wheel', handleWheelZoom, { passive: false });
    return () => document.removeEventListener('wheel', handleWheelZoom);
  }, []);

  // Sync scroll
  const handleScrollA = (e) => {
    if (!isSyncingLeft.current && viewerBRef.current) {
      isSyncingRight.current = true;
      viewerBRef.current.scrollTop = e.target.scrollTop;
      viewerBRef.current.scrollLeft = e.target.scrollLeft;
    }
    isSyncingLeft.current = false;
  };
  const handleScrollB = (e) => {
    if (!isSyncingRight.current && viewerARef.current) {
      isSyncingLeft.current = true;
      viewerARef.current.scrollTop = e.target.scrollTop;
      viewerARef.current.scrollLeft = e.target.scrollLeft;
    }
    isSyncingRight.current = false;
  };

  const getSeverityColor = (s) => {
    if (s === 'High') return 'bg-red-100 text-red-700 border-red-300';
    if (s === 'Medium') return 'bg-amber-100 text-amber-700 border-amber-300';
    return 'bg-sky-100 text-sky-700 border-sky-300';
  };
  const getStatusColor = (s) => {
    if (s === 'New') return 'bg-blue-100 text-blue-700';
    if (s === 'Accepted') return 'bg-emerald-100 text-emerald-700';
    if (s === 'Ignored') return 'bg-gray-200 text-gray-500';
    return 'bg-purple-100 text-purple-700';
  };

  // --- Viewer renderer ---
  const renderDocumentViewer = (title, ref, onScroll, fileState, setter, fileInputRef, dragOver, setDragOver, handleDrop, cacheRef, side) => {
    const pageDiffs = diffs.filter(d => (d.pageA === currentPage || d.pageB === currentPage));

    return (
      <div className="flex-1 flex flex-col min-w-[300px] border-r border-gray-300/60 bg-gray-100">
        <div className="bg-slate-200 p-2 text-xs font-semibold text-slate-700 flex justify-between items-center shadow-sm z-10">
          <span className="flex items-center gap-2">
            <FileText size={14} /> 
            {fileState ? fileState.name : title}
            {fileState && (
              <button onClick={() => { setter(null); setDiffs([]); cacheRef.current = {}; }} className="ml-1 text-slate-400 hover:text-red-500"><X size={12}/></button>
            )}
          </span>
          <span className="text-slate-500 font-normal">Page {currentPage}/{fileState?.totalPages || '?'}</span>
        </div>
        <div 
          ref={ref} onScroll={onScroll}
          onDragOver={(e) => { handleDragOver(e); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className="js-viewer-container flex-1 overflow-auto p-4 flex justify-center items-start relative"
        >
          {!fileState ? (
            <div 
              className={`absolute inset-4 border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-all cursor-pointer
                ${dragOver ? 'border-blue-500 bg-blue-50' : 'border-slate-300 bg-white hover:border-blue-400 hover:bg-slate-50'}`}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={48} className={`mb-4 ${dragOver ? 'text-blue-500' : 'text-slate-300'}`} />
              <p className={`text-base font-semibold mb-1 ${dragOver ? 'text-blue-600' : 'text-slate-500'}`}>
                {dragOver ? 'Thả file vào đây!' : 'Kéo thả file PDF vào đây'}
              </p>
              <p className="text-xs text-slate-400">hoặc click để chọn file</p>
              <input ref={fileInputRef} type="file" accept="application/pdf" className="hidden" onChange={(e) => handleFileSelect(e, setter, cacheRef)} />
            </div>
          ) : (
            <div 
              className="relative"
              style={{ transform: `scale(${zoom / 100})`, transformOrigin: 'top center', transition: 'transform 0.15s ease' }}
            >
              {fileState.pages[currentPage] ? (
                <img src={fileState.pages[currentPage]} alt={`Page ${currentPage}`} className="shadow-lg bg-white block" draggable={false} />
              ) : (
                <div className="w-[800px] h-[1131px] bg-white shadow-lg flex items-center justify-center text-slate-400">Đang render...</div>
              )}

              {/* Diff boxes: In focus mode, only show the selected one */}
              {pageDiffs.map(diff => {
                const isSelected = diff.id === selectedDiffId;
                const isIgnored = diff.status === 'Ignored';
                const isAccepted = diff.status === 'Accepted';
                if (isIgnored && filterStatus !== 'All' && filterStatus !== 'Ignored') return null;
                // Hide accepted diffs from viewer (no more box)
                if (isAccepted) return null;
                // Focus mode: only show selected diff
                if (focusMode && selectedDiffId && !isSelected) return null;

                return (
                  <div
                    key={diff.id}
                    onClick={() => handleSelectDiff(diff.id)}
                    className={`absolute cursor-pointer border-2 transition-all duration-200 ${
                      isSelected 
                        ? 'border-blue-500 bg-blue-400/20 z-20 shadow-[0_0_0_3px_rgba(59,130,246,0.3)]' 
                        : isIgnored
                          ? 'border-gray-400/50 bg-gray-400/5'
                          : 'border-red-500/70 bg-red-500/5 hover:bg-red-400/15'
                    }`}
                    style={{
                      left: `${diff.box.x}%`, top: `${diff.box.y}%`,
                      width: `${diff.box.w}%`, height: `${diff.box.h}%`,
                    }}
                  >
                    <div className={`absolute -top-5 left-0 text-white text-[9px] px-1.5 py-0.5 rounded whitespace-nowrap shadow ${isSelected ? 'bg-blue-600' : 'bg-red-600/80'}`}>
                      #{diff.id.split('-')[1]}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    );
  };

  const getPageDiffCount = (p) => diffs.filter(d => d.pageA === p || d.pageB === p).length;

  return (
    <div className="flex flex-col h-screen bg-slate-50 text-sm font-sans">
      
      {/* TOOLBAR */}
      <header className="bg-white border-b border-gray-300 px-4 py-2 flex items-center justify-between shadow-sm z-20">
        <div className="flex items-center gap-3">
          <div className="font-bold text-lg text-slate-800 tracking-tight flex items-center gap-2">
            <Layers className="text-blue-600" /> PDF Comparator
          </div>
          <div className="h-6 w-px bg-gray-300"></div>
          
          <button onClick={() => fileInputARef.current?.click()} className={`flex items-center gap-1.5 px-3 py-1.5 border rounded transition-colors text-xs ${fileA ? 'bg-green-50 border-green-300 text-green-700' : 'bg-slate-100 border-slate-300 text-slate-700 hover:bg-slate-200'}`}>
            <FileText size={13} /> {fileA ? fileA.name : 'File A...'}
          </button>
          <button onClick={() => fileInputBRef.current?.click()} className={`flex items-center gap-1.5 px-3 py-1.5 border rounded transition-colors text-xs ${fileB ? 'bg-green-50 border-green-300 text-green-700' : 'bg-slate-100 border-slate-300 text-slate-700 hover:bg-slate-200'}`}>
            <FileText size={13} /> {fileB ? fileB.name : 'File B...'}
          </button>
          <input ref={fileInputARef} type="file" accept="application/pdf" className="hidden" onChange={(e) => handleFileSelect(e, setFileA, canvasCacheA)} />
          <input ref={fileInputBRef} type="file" accept="application/pdf" className="hidden" onChange={(e) => handleFileSelect(e, setFileB, canvasCacheB)} />
          
          <button onClick={openMappingSetup} disabled={isComparing || !fileA || !fileB}
            className={`flex items-center gap-2 px-4 py-1.5 rounded font-medium text-white transition-colors ${isComparing || !fileA || !fileB ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}`}>
            <Play size={14} fill="currentColor" />
            {isComparing ? 'Đang phân tích...' : 'Compare...'}
          </button>

          {diffs.length > 0 && (
            <>
              <div className="h-6 w-px bg-gray-300"></div>
              <button onClick={() => navigateDiff('prev')} className="p-1 text-slate-600 hover:bg-slate-200 rounded" title="Prev (P)"><ChevronLeft size={18}/></button>
              <span className="text-xs text-slate-500 font-medium min-w-[40px] text-center">
                {selectedDiffId ? `${filteredDiffs.findIndex(d => d.id === selectedDiffId) + 1}/${filteredDiffs.length}` : `-/${filteredDiffs.length}`}
              </span>
              <button onClick={() => navigateDiff('next')} className="p-1 text-slate-600 hover:bg-slate-200 rounded" title="Next (N)"><ChevronRight size={18}/></button>
              
              <div className="h-6 w-px bg-gray-300"></div>
              <button 
                onClick={() => setFocusMode(!focusMode)} 
                className={`p-1.5 rounded text-xs flex items-center gap-1 ${focusMode ? 'bg-blue-100 text-blue-700' : 'text-slate-500 hover:bg-slate-200'}`}
                title={focusMode ? 'Focus Mode ON: Chỉ hiện diff đang chọn' : 'Focus Mode OFF: Hiện tất cả diff'}
              >
                {focusMode ? <Maximize2 size={14}/> : <Eye size={14}/>}
                {focusMode ? 'Focus' : 'All'}
              </button>
            </>
          )}
        </div>

        <div className="flex items-center gap-3">
          {(fileA || fileB) && (
            <div className="flex items-center gap-1 bg-slate-100 rounded border border-slate-300 p-0.5">
              <button onClick={goToPrevPage} disabled={currentPage <= 1} className="p-1 hover:bg-white rounded disabled:opacity-30"><ChevronLeft size={16}/></button>
              <span className="px-1.5 text-xs font-medium">{currentPage}/{maxPages}</span>
              <button onClick={goToNextPage} disabled={currentPage >= maxPages} className="p-1 hover:bg-white rounded disabled:opacity-30"><ChevronRight size={16}/></button>
            </div>
          )}
          <div className="flex items-center bg-slate-100 rounded border border-slate-300 p-0.5">
            <button onClick={() => setZoom(Math.max(20, zoom - 20))} className="p-1 hover:bg-white rounded"><ZoomOut size={16}/></button>
            <span className="px-1.5 w-11 text-center text-xs font-medium">{zoom}%</span>
            <button onClick={() => setZoom(Math.min(400, zoom + 20))} className="p-1 hover:bg-white rounded"><ZoomIn size={16}/></button>
          </div>
          <button className="p-1.5 text-slate-600 hover:bg-slate-200 rounded"><Settings size={16}/></button>
          <button className="p-1.5 text-slate-600 hover:bg-slate-200 rounded"><Save size={16}/></button>
          <button className="p-1.5 text-slate-600 hover:bg-slate-200 rounded"><Download size={16}/></button>
        </div>
      </header>

      {/* MAIN - vertical split: top = viewers+list, bottom = detail panel */}
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* TOP AREA */}
        <div className="flex-1 flex overflow-hidden" style={{ minHeight: '200px' }}>
        
          {/* FILMSTRIP */}
          <aside className="w-14 bg-slate-100 border-r border-gray-300 flex flex-col items-center py-3 gap-2 z-10 overflow-y-auto">
            <div className="text-[9px] font-bold text-slate-400 uppercase mb-1">Pg</div>
            {Array.from({ length: maxPages }, (_, i) => i + 1).map(p => {
              const cnt = getPageDiffCount(p);
              return (
                <div key={p} onClick={() => setCurrentPage(p)}
                  className={`w-9 h-12 bg-white shadow-sm relative cursor-pointer transition-all hover:scale-105 ${currentPage === p ? 'border-2 border-blue-500 ring-2 ring-blue-200' : 'border border-gray-300'}`}>
                  <span className="absolute inset-0 flex items-center justify-center text-[9px] text-slate-400">{p}</span>
                  {cnt > 0 && (
                    <div className="absolute -top-1 -right-1 bg-red-500 text-white text-[7px] w-3 h-3 rounded-full flex items-center justify-center font-bold">{cnt}</div>
                  )}
                </div>
              );
            })}
          </aside>

          {/* VIEWERS */}
          <section className="flex-1 flex bg-gray-300 relative">
            {isComparing && (
              <div className="absolute inset-0 bg-white/85 z-50 flex flex-col items-center justify-center backdrop-blur-sm">
                <div className="w-72 h-2 bg-gray-200 rounded-full overflow-hidden mb-3">
                  <div className="h-full bg-blue-600 rounded-full transition-all duration-200" style={{ width: `${progress}%` }}></div>
                </div>
                <span className="text-slate-600 font-medium text-sm">{progressText}</span>
              </div>
            )}
            {renderDocumentViewer("Kéo thả File A", viewerARef, handleScrollA, fileA, setFileA, fileInputARef, dragOverA, setDragOverA, handleDropA, canvasCacheA, 'A')}
            {renderDocumentViewer("Kéo thả File B", viewerBRef, handleScrollB, fileB, setFileB, fileInputBRef, dragOverB, setDragOverB, handleDropB, canvasCacheB, 'B')}
          </section>

          {/* ISSUE LIST SIDEBAR */}
          <aside style={{ width: `${sidebarWidth}px`, minWidth: '200px', maxWidth: '500px' }} className="bg-white border-l border-gray-300 flex flex-col z-10 relative">
            <div 
              className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize hover:bg-blue-400/40 active:bg-blue-500/50 z-30"
              onMouseDown={(e) => {
                e.preventDefault();
                isResizing.current = true;
                const startX = e.clientX;
                const startW = sidebarWidth;
                const onMove = (ev) => { if (!isResizing.current) return; setSidebarWidth(Math.max(200, Math.min(500, startW + startX - ev.clientX))); };
                const onUp = () => { isResizing.current = false; window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
                window.addEventListener('mousemove', onMove);
                window.addEventListener('mouseup', onUp);
              }}
            />
            
            <div className="p-2.5 border-b border-gray-200 bg-slate-50 flex justify-between items-center">
              <h3 className="font-semibold text-slate-800 flex items-center gap-2 text-sm"><List size={15}/> Issues</h3>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${diffs.length > 0 ? 'bg-red-100 text-red-700' : 'bg-slate-200 text-slate-500'}`}>{filteredDiffs.length}/{diffs.length}</span>
            </div>

            <div className="p-1.5 border-b border-gray-200 flex gap-1.5 items-center">
              <select className="bg-slate-50 border border-slate-300 text-slate-700 text-[11px] rounded px-1.5 py-1 flex-1 outline-none"
                value={filterStatus} onChange={(e) => setFilterStatus(e.target.value)}>
                <option value="All">Tất cả ({diffs.length})</option>
                <option value="New">Mới ({diffs.filter(d=>d.status==='New').length})</option>
                <option value="Accepted">Duyệt ({diffs.filter(d=>d.status==='Accepted').length})</option>
                <option value="Ignored">Bỏ qua ({diffs.filter(d=>d.status==='Ignored').length})</option>
              </select>
              <button 
                onClick={() => setHideAccepted(!hideAccepted)}
                className={`p-1 rounded border transition-colors ${hideAccepted ? 'bg-emerald-100 border-emerald-300 text-emerald-700' : 'bg-slate-50 border-slate-300 text-slate-500'}`}
                title={hideAccepted ? 'Đang ẩn Accepted' : 'Ẩn Accepted'}
              >
                {hideAccepted ? <EyeOff size={12}/> : <Eye size={12}/>}
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-1.5 space-y-1">
              {diffs.length === 0 && !isComparing && (
                <div className="text-center text-slate-400 mt-8 p-4">
                  <LayoutTemplate size={24} className="mx-auto mb-2 opacity-40" />
                  <p className="text-[11px]">{(!fileA || !fileB) ? 'Chọn 2 file PDF' : 'Nhấn Compare'}</p>
                </div>
              )}
              {filteredDiffs.map(diff => (
                <div key={diff.id} onClick={() => handleSelectDiff(diff.id)}
                  className={`p-2 rounded border text-left cursor-pointer transition-all ${
                    selectedDiffId === diff.id 
                      ? 'border-blue-500 bg-blue-50 shadow ring-1 ring-blue-400' 
                      : diff.status === 'Ignored' ? 'border-gray-200 bg-gray-50 opacity-50'
                      : diff.status === 'Accepted' ? 'border-emerald-200 bg-emerald-50/50'
                      : 'border-slate-200 bg-white hover:border-blue-300'
                  }`}>
                  <div className="flex justify-between items-center mb-0.5">
                    <div className="flex items-center gap-1">
                      <span className="text-[10px] font-bold text-slate-500">#{diff.id.split('-')[1]}</span>
                      <span className={`text-[8px] uppercase font-bold px-1 py-0.5 rounded border ${getSeverityColor(diff.severity)}`}>{diff.severity}</span>
                    </div>
                    <span className={`text-[8px] font-medium px-1.5 py-0.5 rounded-full ${getStatusColor(diff.status)}`}>{diff.status}</span>
                  </div>
                  <p className="text-[10px] text-slate-600 leading-tight truncate">{diff.description}</p>
                </div>
              ))}
            </div>

            {diffs.length > 0 && (
              <div className="p-1.5 border-t border-gray-200 text-[9px] text-slate-400 bg-slate-50 text-center">
                <kbd className="bg-slate-200 px-1 rounded">N</kbd>/<kbd className="bg-slate-200 px-1 rounded">P</kbd> · 
                <kbd className="bg-slate-200 px-1 rounded ml-1">A</kbd> Accept · 
                <kbd className="bg-slate-200 px-1 rounded ml-1">I</kbd> Ignore
              </div>
            )}
          </aside>
        </div>

        {/* BOTTOM DETAIL PANEL */}
        {selectedDiff && (
          <div style={{ height: `${detailPanelHeight}px` }} className="js-detail-panel bg-slate-50 border-t-2 border-gray-300 flex flex-col relative shrink-0">
            {/* Resize handle */}
            <div 
              className="absolute left-0 right-0 top-0 h-1.5 cursor-row-resize hover:bg-blue-400/40 active:bg-blue-500/50 z-30"
              onMouseDown={(e) => {
                e.preventDefault();
                const startY = e.clientY;
                const startH = detailPanelHeight;
                const onMove = (ev) => setDetailPanelHeight(Math.max(120, Math.min(window.innerHeight * 0.6, startH + startY - ev.clientY)));
                const onUp = () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
                window.addEventListener('mousemove', onMove);
                window.addEventListener('mouseup', onUp);
              }}
            />

            {/* Header */}
            <div className="flex items-center justify-between px-4 py-1.5 border-b border-gray-200 bg-white shrink-0">
              <div className="flex items-center gap-3">
                <span className="text-sm font-bold text-slate-800">#{selectedDiff.id.split('-')[1]} — Chi tiết thay đổi</span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded border font-bold ${getSeverityColor(selectedDiff.severity)}`}>{selectedDiff.severity}</span>
                <span className="text-[10px] text-slate-400">Trang {selectedDiff.pageA} · {selectedDiff.pixelCount}px</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-0.5 bg-slate-100 rounded border border-slate-300 p-0.5">
                  <button onClick={() => setCropZoom(Math.max(50, cropZoom - 50))} className="p-0.5 hover:bg-white rounded"><ZoomOut size={13}/></button>
                  <span className="px-1 text-[10px] font-medium min-w-[28px] text-center">{cropZoom}%</span>
                  <button onClick={() => setCropZoom(Math.min(500, cropZoom + 50))} className="p-0.5 hover:bg-white rounded"><ZoomIn size={13}/></button>
                </div>
                <button onClick={() => handleStatusChange(selectedDiff.id, 'Accepted')} className="px-3 py-1 text-xs font-semibold text-emerald-700 bg-emerald-100 hover:bg-emerald-200 rounded-lg flex items-center gap-1"><CheckCircle size={13}/> Accept</button>
                <button onClick={() => handleStatusChange(selectedDiff.id, 'Ignored')} className="px-3 py-1 text-xs font-semibold text-gray-600 bg-gray-200 hover:bg-gray-300 rounded-lg flex items-center gap-1"><EyeOff size={13}/> Ignore</button>
                <button onClick={() => setSelectedDiffId(null)} className="p-1 text-slate-400 hover:text-slate-600 rounded"><X size={16}/></button>
              </div>
            </div>

            {/* Crop previews side-by-side */}
            <div className="flex-1 flex gap-2 p-2 overflow-hidden min-h-0">
              <div className="flex-1 flex flex-col min-w-0">
                <div className="text-xs text-red-600 font-semibold mb-1 flex items-center gap-1 px-1 shrink-0">
                  <span className="w-2 h-2 rounded-full bg-red-500"></span> File A (Bản cũ)
                </div>
                <div className="flex-1 overflow-auto border-2 border-red-200 rounded-lg bg-slate-200 min-h-0 text-center">
                  {selectedDiff.cropA ? (
                    <img src={selectedDiff.cropA} alt="A" style={{ width: `${cropZoom}%`, maxWidth: 'none', transition: 'width 0.1s', display: 'inline-block' }} draggable={false} />
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-slate-400">N/A</div>
                  )}
                </div>
              </div>
              <div className="flex-1 flex flex-col min-w-0">
                <div className="text-xs text-blue-600 font-semibold mb-1 flex items-center gap-1 px-1 shrink-0">
                  <span className="w-2 h-2 rounded-full bg-blue-500"></span> File B (Bản mới)
                </div>
                <div className="flex-1 overflow-auto border-2 border-blue-200 rounded-lg bg-slate-200 min-h-0 text-center">
                  {selectedDiff.cropB ? (
                    <img src={selectedDiff.cropB} alt="B" style={{ width: `${cropZoom}%`, maxWidth: 'none', transition: 'width 0.1s', display: 'inline-block' }} draggable={false} />
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-slate-400">N/A</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* STATUS BAR */}
      <footer className="bg-slate-800 text-slate-300 text-xs px-4 py-1 flex justify-between items-center z-20">
        <span>{isComparing ? progressText : (diffs.length > 0 ? `${diffs.filter(d=>d.status==='New').length} mới · ${diffs.filter(d=>d.status==='Accepted').length} duyệt · ${diffs.filter(d=>d.status==='Ignored').length} bỏ qua` : 'Ready')}</span>
        <span>Trang {currentPage}/{maxPages} · Scale {renderScale}x · {focusMode ? 'Focus' : 'All'}</span>
      </footer>

      {/* SETUP MAPPING MODAL */}
      {isSetupOpen && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-8 backdrop-blur-sm">
          <div className="bg-white rounded-xl shadow-2xl w-[600px] flex flex-col max-h-[85vh] overflow-hidden">
            <div className="p-4 border-b border-gray-200 bg-slate-50 flex justify-between items-center">
              <div>
                <h2 className="text-lg font-bold text-slate-800">Cấu hình ghép cặp trang (Page Mapping)</h2>
                <p className="text-xs text-slate-500">Hệ thống đã phân tích tự động. Bạn có thể điều chỉnh nếu thấy sai lệch.</p>
              </div>
              <button onClick={() => setIsSetupOpen(false)} className="text-slate-400 hover:text-red-500"><X size={20}/></button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 bg-slate-100">
              <div className="flex bg-slate-200 font-bold text-xs text-slate-600 rounded-t border border-gray-300">
                <div className="flex-1 px-3 py-2 border-r border-gray-300 text-center text-red-700 bg-red-50/50">File A (Bản cũ)</div>
                <div className="w-16 px-2 py-2 text-center text-slate-400">Match</div>
                <div className="flex-1 px-3 py-2 border-l border-gray-300 text-center text-blue-700 bg-blue-50/50">File B (Bản mới)</div>
              </div>
              
              <div className="border border-t-0 border-gray-300 rounded-b bg-white divide-y divide-gray-200">
                {proposedMapping.map((map, idx) => (
                  <div key={idx} className="flex items-center text-sm">
                    <div className="flex-1 px-3 py-2 text-center text-slate-700 font-medium">
                      {map.pageA ? `Trang ${map.pageA}` : <span className="text-slate-400 italic">Trống</span>}
                    </div>
                    <div className="w-16 flex justify-center text-slate-300">
                      {map.type === 'added' || map.type === 'removed' ? <span className="text-red-400 text-xs font-bold">X</span> : <span className="text-emerald-500">↔</span>}
                    </div>
                    <div className="flex-1 px-3 py-2 text-center">
                      <select 
                        className="w-full bg-slate-50 border border-slate-300 rounded px-2 py-1 text-sm text-slate-700 font-medium"
                        value={map.pageB || ''}
                        onChange={(e) => {
                          const val = e.target.value;
                          const newMap = [...proposedMapping];
                          if (val === '') {
                            newMap[idx].pageB = null;
                            newMap[idx].type = newMap[idx].pageA ? 'removed' : 'none'; // Edge case
                          } else {
                            newMap[idx].pageB = parseInt(val, 10);
                            newMap[idx].type = 'modified'; // Assume modified upon manual choice
                          }
                          setProposedMapping(newMap);
                        }}
                      >
                        <option value="">-- Trống (Xóa) --</option>
                        {Array.from({ length: fileB?.totalPages || Math.max(1, map.pageB || 1) }, (_, i) => (
                          <option key={i+1} value={i+1}>Trang {i+1}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="p-4 border-t border-gray-200 bg-white flex justify-between items-center gap-3">
              <span className="text-xs text-slate-500 italic flex-1">Ghi chú: Nếu file thêm bớt trang, các mũi tên ↔ sẽ tự động nhận diện.</span>
              <button onClick={() => setIsSetupOpen(false)} className="px-4 py-2 border rounded-lg text-sm font-semibold text-slate-600 hover:bg-slate-50">Hủy</button>
              <button 
                onClick={executeCompare} 
                className="px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-bold shadow hover:bg-blue-700 flex items-center gap-2"
              >
                <CheckCircle size={16}/> Bắt đầu So sánh
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

