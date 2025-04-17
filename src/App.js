// File: src/App.js
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import './App.css';

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;
// Log the version being used
console.log(`Using pdf.js version: ${pdfjs.version}`);

function App() {
  const backendUrl = process.env.REACT_APP_BACKEND_URL;
  console.log('Backend URL in App component:', backendUrl);
  
  // State to manage which view is shown: 'selector' or 'chatbot'
  const [currentView, setCurrentView] = useState('selector');
  // State to hold the name of the currently selected/active document set
  const [currentSetName, setCurrentSetName] = useState(null);

  // Function called by SetSelector when a set is chosen or created
  const handleSetSelected = (setName) => {
    console.log(`Set selected/created: ${setName}`);
    setCurrentSetName(setName);
    setCurrentView('chatbot'); // Switch to chatbot view
  };

  // Function called by Chatbot to go back to the selector
  const handleBackToSelector = () => {
    setCurrentSetName(null); // Clear the current set
    setCurrentView('selector'); // Switch back to selector view
  };

  // Render the appropriate component based on the current view
  return (
    <div className="app-root"> {/* Add a root container if needed */} 
      {currentView === 'selector' ? (
        <SetSelector 
          onSetSelected={handleSetSelected} 
      backendUrl={backendUrl}
    />
      ) : currentSetName ? ( // Only show chatbot if a set name is selected
    <Chatbot 
          setName={currentSetName} // Pass setName prop
          onBack={handleBackToSelector}
      backendUrl={backendUrl}
    />
      ) : (
        // Fallback case, should ideally not happen if logic is correct
        <div>Error: No set selected. <button onClick={handleBackToSelector}>Go Back</button></div> 
      )}
    </div>
  );
}

// --- Set Selector Component (NEW) ---
function SetSelector({ onSetSelected, backendUrl }) {
  const [setList, setSetList] = useState([]);
  const [isLoadingList, setIsLoadingList] = useState(true);
  const [listError, setListError] = useState('');
  
  const [newSetName, setNewSetName] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isCreating, setIsCreating] = useState(false);
  const [createError, setCreateError] = useState('');
  const [createStatus, setCreateStatus] = useState(''); // For creation progress/status

  // State for deletion status
  const [isDeleting, setIsDeleting] = useState(null); // Store name of set being deleted
  const [deleteError, setDeleteError] = useState('');
  const [deleteStatus, setDeleteStatus] = useState('');
  
  const fileInputRef = useRef(null);
  
  // Fetch existing sets on component mount
  const fetchSets = useCallback(async () => {
    setIsLoadingList(true);
    setListError('');
    try {
      const response = await fetch(`${backendUrl}/document_sets`);
      if (!response.ok) {
        throw new Error(`Failed to fetch sets: ${response.statusText}`);
      }
      const data = await response.json();
      setSetList(data.set_names || []);
    } catch (err) {
      console.error('Error fetching document sets:', err);
      setListError(`Failed to load existing sets: ${err.message}`);
      setSetList([]); // Ensure list is empty on error
    } finally {
      setIsLoadingList(false);
    }
  }, [backendUrl]);

  useEffect(() => {
    fetchSets();
  }, [fetchSets]);

  // Handle file selection for new set
  const handleFileChange = (e) => {
    const pdfFiles = Array.from(e.target.files).filter(
      file => file.type === 'application/pdf'
    );
    if (pdfFiles.length > 0) {
      setSelectedFiles(prevFiles => [...prevFiles, ...pdfFiles]);
      setCreateError(''); // Clear error when files are selected
    } else if (e.target.files.length > 0) {
        setCreateError('Please select PDF files only.');
    }
    // Reset file input to allow selecting the same file again
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  
  // Remove file from the selected list
  const handleRemoveFile = (index) => {
    setSelectedFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
  };

  // Handle creation of a new set
  const handleCreateSet = async () => {
    if (!newSetName.trim()) {
      setCreateError('Please enter a name for the document set.');
      return;
    }
    if (selectedFiles.length === 0) {
      setCreateError('Please select at least one PDF file to upload.');
      return;
    }
    
    setIsCreating(true);
    setCreateError('');
    setCreateStatus('Uploading and processing files...');

    const formData = new FormData();
    formData.append('name', newSetName.trim());
    selectedFiles.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch(`${backendUrl}/document_sets`, {
        method: 'POST',
        body: formData,
        // No 'Content-Type' header needed for FormData, browser sets it with boundary
      });

      const responseData = await response.json(); // Attempt to parse JSON regardless of status

      if (!response.ok) {
         // Use detail from JSON response if available, otherwise statusText
         const errorDetail = responseData.detail || response.statusText;
         throw new Error(`Failed to create set: ${response.status} ${errorDetail}`);
      }

      setCreateStatus(`Set '${responseData.set_name}' created successfully!`);
      fetchSets(); // Refresh the list after creation
      
      // Wait a moment before switching view
      setTimeout(() => {
         onSetSelected(responseData.set_name); // Notify parent component
      }, 1500); // Delay to show success message

    } catch (err) {
      console.error('Error creating document set:', err);
      setCreateError(`Creation failed: ${err.message}`);
      setCreateStatus(''); // Clear status on error
    } finally {
      setIsCreating(false);
      // Optionally clear form on success/failure after a delay?
      // setNewSetName('');
      // setSelectedFiles([]);
    }
  };

  // --- Handle Deletion of a Set --- (NEW)
  const handleDeleteSet = async (setNameToDelete) => {
    if (!window.confirm(`Are you sure you want to permanently delete the document set "${setNameToDelete}"? This cannot be undone.`)) {
      return; // User cancelled
    }

    setIsDeleting(setNameToDelete); // Set deleting status for this specific set
    setDeleteError('');
    setDeleteStatus('');

    try {
        const encodedSetName = encodeURIComponent(setNameToDelete); // Ensure name is URL-safe
        const response = await fetch(`${backendUrl}/document_sets/${encodedSetName}`, {
            method: 'DELETE',
        });

        const responseData = await response.json(); // Try to get response body

        if (!response.ok) {
            const errorDetail = responseData.detail || response.statusText;
            throw new Error(`Failed to delete set: ${response.status} ${errorDetail}`);
        }

        setDeleteStatus(`Set '${setNameToDelete}' deleted successfully.`);
        // Update the list visually by removing the deleted set
        setSetList(prevList => prevList.filter(name => name !== setNameToDelete));
        // Clear success message after a delay
        setTimeout(() => setDeleteStatus(''), 3000); 
      
    } catch (err) {
        console.error(`Error deleting document set '${setNameToDelete}':`, err);
        setDeleteError(`Failed to delete '${setNameToDelete}': ${err.message}`);
        // Clear error message after a delay
        setTimeout(() => setDeleteError(''), 5000);
    } finally {
        setIsDeleting(null); // Reset deleting status
    }
  };
  
  // Render the Set Selector UI
  return (
    <div className="set-selector-container"> {/* Updated class name */} 
      <div className="set-selector-header"> {/* Updated class name */} 
        <h1>Document Chatbot</h1>
        <p>Select an existing document set or create a new one</p>
      </div>
      
      <div className="set-selector-content"> {/* Updated class name */} 
        {/* Section for Existing Sets */}
        <div className="existing-sets-section"> {/* Updated class name */} 
          <h2>Existing Document Sets</h2>
          {/* Display Deletion Status/Error */}
          {deleteStatus && <div className="status-message success">{deleteStatus}</div>}
          {deleteError && <div className="error-message delete-error">{deleteError}</div>}
          {isLoadingList ? (
            <p>Loading sets...</p>
          ) : listError ? (
            <div className="error-message">{listError}</div>
          ) : setList.length === 0 ? (
            <p>No existing document sets found.</p>
          ) : (
            <ul className="set-list"> {/* Updated class name */} 
              {setList.map(setName => (
                <li key={setName} className="set-list-item"> {/* Updated class name */} 
                  <span>{setName}</span>
                  <div className="set-item-buttons">
                    <button 
                      onClick={() => onSetSelected(setName)}
                      disabled={isDeleting === setName} // Disable if being deleted
                    >
                      Select
                    </button>
                    <button 
                      className="delete-button" 
                      onClick={() => handleDeleteSet(setName)}
                      disabled={isDeleting === setName} // Disable if being deleted
                      title={`Delete set '${setName}'`}
                    >
                       {isDeleting === setName ? 'Deleting...' : 'Delete'}
                    </button>
                   </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Divider */}
        <hr className="set-selector-divider" /> {/* Updated class name */} 

        {/* Section for Creating a New Set */}
        <div className="create-set-section"> {/* Updated class name */} 
          <h2>Create New Document Set</h2>
          <div className="create-set-form"> {/* Updated class name */} 
            <input
              type="text"
              placeholder="Enter name for new set..."
              value={newSetName}
              onChange={(e) => setNewSetName(e.target.value)}
              disabled={isCreating}
              className="set-name-input" /* Optional: specific class */
            />

            <div className="file-drop-area-small" /* Optional: smaller drop area */
                 onClick={() => !isCreating && fileInputRef.current?.click()}> 
          <input
            type="file"
            accept=".pdf"
            multiple
            onChange={handleFileChange}
            ref={fileInputRef}
            style={{ display: 'none' }}
                disabled={isCreating}
          />
              <div className="drop-message-small"> {/* Optional: smaller message */} 
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
                <p>{selectedFiles.length > 0 ? `${selectedFiles.length} file(s) selected` : 'Click or drag PDF files here'}</p>
          </div>
        </div>
        
            {selectedFiles.length > 0 && (
              <div className="file-list-small"> {/* Optional: smaller list */} 
                {/* <h4>Selected Files:</h4> */}
                <ul>
                  {selectedFiles.map((file, index) => (
                    <li key={index} className="file-item-small"> {/* Optional: smaller item */} 
                  <span className="file-name">{file.name}</span>
                  <span className="file-size">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                  <button 
                    className="remove-file-btn"
                    onClick={() => handleRemoveFile(index)}
                        disabled={isCreating}
                  >
                    &times;
                  </button>
                </li>
              ))}
            </ul>
              </div>
            )}
            
            <button 
              className="create-set-btn" /* Optional: specific class */
              onClick={handleCreateSet}
              disabled={isCreating || !newSetName.trim() || selectedFiles.length === 0}
            >
              {isCreating ? 'Creating...' : 'Create Document Set'}
            </button>
            
            {isCreating && <div className="status-message">{createStatus}</div>}
            {createError && <div className="error-message create-error">{createError}</div>} 
            {!isCreating && createStatus && <div className="status-message success">{createStatus}</div>} 

            </div>
          </div>
      </div>
    </div>
  );
}


// --- Chatbot Component (UPDATED) ---
function Chatbot({ setName, onBack, backendUrl }) { // Accept setName prop
  console.log(`Using pdf.js version: ${pdfjs.version} in Chatbot component`);
  console.log('Backend URL in Chatbot component:', backendUrl);
  console.log('Current Document Set Name:', setName); // Log setName
  
  // ==================================
  // 1. STATE (mostly unchanged)
  // ==================================
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentPdf, setCurrentPdf] = useState(null);
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [scale, setScale] = useState(1.5);
  const [error, setError] = useState('');
  const [pdfLoadingStatus, setPdfLoadingStatus] = useState('idle');
  const [currentChunk, setCurrentChunk] = useState(null);
  const [highlightAreas, setHighlightAreas] = useState([]);
  const [pdfProxy, setPdfProxy] = useState(null);
  const [currentPdfOriginalUrl, setCurrentPdfOriginalUrl] = useState(null);
  const [showDebugInfo, setShowDebugInfo] = useState(false);
  const [showChunkBrowser, setShowChunkBrowser] = useState(false);
  const [allChunks, setAllChunks] = useState([]); 
  const [lastRenderedPage, setLastRenderedPage] = useState(null);

  // ==================================
  // 2. REFS (unchanged)
  // ==================================
  const messagesEndRef = useRef(null);
  const pdfViewerRef = useRef(null);
  const optionsRef = useRef({}); 

  // ==================================
  // 3. DERIVED STATE (unchanged)
  // ==================================
  const safePageNumber = Math.min(Math.max(1, pageNumber), numPages || 1);

  // ==================================
  // 4. HELPER FUNCTIONS (Callbacks)
  // ==================================

  // --- PDF Loading (fetchPdf - Check URL logic) ---
  const fetchPdf = useCallback(async (url, targetPage, chunk) => {
    setPdfLoadingStatus('loading');
      setHighlightAreas([]); 
      console.log("Attempting to load PDF:", url, "Target page:", targetPage);
    
    try {
      let pdfUrl = url;
          // If URL doesn't start with http or blob, assume it's relative to backend
          if (pdfUrl && !pdfUrl.startsWith('http') && !pdfUrl.startsWith('blob:')) {
              // **Important**: The URL from the backend should already be absolute or correctly relative.
              // If it's relative like "/document_sets/...", prepend backendUrl.
              // If it's already absolute (http://...), use it directly.
              // Let's assume backend sends URLs starting with /document_sets/... if relative
              if (pdfUrl.startsWith('/')) { 
        pdfUrl = `${backendUrl}${url}`;
              } else {
                   // If it doesn't start with /, it might be a malformed URL from backend
                   console.warn("Received potentially malformed relative PDF URL:", url);
                   // Attempt to prepend anyway, or handle error
                   pdfUrl = `${backendUrl}/${url}`; // Best guess
              }
          }
          console.log("Final PDF URL for fetching:", pdfUrl);

          // Cleanup previous blob URL
          if (currentPdf && currentPdf.startsWith('blob:')) {
              console.log("Revoking previous blob URL:", currentPdf);
              URL.revokeObjectURL(currentPdf);
          }

          // XHR Request (remains the same)
      const xhr = new XMLHttpRequest();
      xhr.open('GET', pdfUrl, true);
      xhr.responseType = 'arraybuffer';
          xhr.withCredentials = true; // Keep if needed for auth/cookies
      
      const pdfPromise = new Promise((resolve, reject) => {
               // ... (XHR onload, onerror, ontimeout remain the same) ...
        xhr.onload = function() {
          if (this.status === 200) {
            const pdfData = new Uint8Array(this.response);
            const blob = new Blob([pdfData], { type: 'application/pdf' });
            const blobUrl = URL.createObjectURL(blob);
            resolve(blobUrl);
          } else {
                        reject(new Error(`XHR failed with status ${this.status}: ${this.statusText}`));
          }
        };
        xhr.onerror = function() {
                    reject(new Error('XHR request failed (network error)'));
                };
                xhr.ontimeout = function () {
                    reject(new Error('XHR request timed out'));
        };
      });
      
      xhr.send();
      const blobUrl = await pdfPromise;

          // Reset state for new PDF
          setCurrentPdf(blobUrl); 
          setCurrentPdfOriginalUrl(url); // Store the ORIGINAL url from the chunk for comparison
          setNumPages(null);
          setPdfProxy(null); 
          setPageNumber(targetPage || 1); 
      setCurrentChunk(chunk);
          setError(''); 
      
    } catch (error) {
      console.error("Error fetching PDF:", error);
      setError(`Could not load the PDF: ${error.message}`);
          setCurrentPdf(null);
          setCurrentPdfOriginalUrl(null);
          setNumPages(null);
          setPdfProxy(null);
      setPdfLoadingStatus('error');
      }
  // Dependencies: Added backendUrl if used in URL construction
  }, [backendUrl, currentPdf]); // Removed setters, kept core state/props

  // --- Scrolling (scrollToHighlight - unchanged) ---
  const scrollToHighlight = useCallback(/* ... unchanged ... */ (highlight) => {
      if (!pdfViewerRef.current || !highlight) return;
      const container = pdfViewerRef.current;
    const highlightCenterY = highlight.top + (highlight.height / 2);
      const containerHeight = container.clientHeight;
      const containerScrollTop = container.scrollTop;
    const containerScrollBottom = containerScrollTop + containerHeight;
    
      // Add some padding - don't scroll if it's reasonably visible
      const visibilityPadding = 50; // pixels
      if (highlightCenterY >= containerScrollTop + visibilityPadding && highlightCenterY <= containerScrollBottom - visibilityPadding) {
          console.log("HIGHLIGHT_DEBUG: Highlight already sufficiently visible, skipping scroll.");
          return;
    }
    
    // Calculate target scroll position to center the highlight
      const targetScrollTop = Math.max(0, highlightCenterY - (containerHeight / 2)); // Ensure not scrolling < 0
    
      console.log("HIGHLIGHT_DEBUG: Scrolling to highlight", { targetScrollTop });
      container.scrollTo({
      top: targetScrollTop,
      behavior: 'smooth'
    });
  }, [pdfViewerRef]);

  // --- Coordinate Transformation (transformCoordinates - unchanged) ---
  const transformCoordinates = useCallback(/* ... unchanged ... */ async () => {
    console.log("HIGHLIGHT_DEBUG: ======= Attempting transformCoordinates ======");
    if (!currentChunk || !currentChunk.position) {
      console.log("HIGHLIGHT_DEBUG: Exiting - Missing currentChunk or position data.");
      setHighlightAreas([]);
      return;
    }
    if (!pdfProxy) {
        console.log("HIGHLIGHT_DEBUG: Exiting - pdfProxy is not available yet.");
        setHighlightAreas([]);
      return;
    }
    
    const pageElement = document.querySelector('.react-pdf__Page, .pdf-page-rendered');
    const container = pdfViewerRef.current;
    
    if (!pageElement || !container) {
      console.log("HIGHLIGHT_DEBUG: Exiting - Page element or container not found.");
      setHighlightAreas([]);
      return;
    }
    
    try {
      const position = currentChunk.position;
      console.log("HIGHLIGHT_DEBUG: Starting transformation with position:", JSON.stringify(position));

      // Ensure bbox exists and has properties (like 'l') before using
      if (!position.bbox || typeof position.bbox.l !== 'number' || typeof position.bbox.t !== 'number' || typeof position.bbox.r !== 'number' || typeof position.bbox.b !== 'number') {
        console.log("HIGHLIGHT_DEBUG: Exiting - Invalid or missing bbox data.", position.bbox);
        setHighlightAreas([]);
        return;
      }
      const bbox = position.bbox;
      const coordOrigin = position.coord_origin || 'BOTTOMLEFT';

      let pdfPageWidth, pdfPageHeight;
      try {
          const page = await pdfProxy.getPage(safePageNumber);
          const viewport = page.getViewport({ scale: 1 });
          pdfPageWidth = viewport.width;
          pdfPageHeight = viewport.height;
          if (!pdfPageWidth || !pdfPageHeight) throw new Error("getViewport returned invalid dimensions");
          console.log("HIGHLIGHT_DEBUG: Got viewport dimensions", { pdfPageWidth, pdfPageHeight });
      } catch (pageError) {
          console.error("HIGHLIGHT_DEBUG: Error getting page or viewport:", pageError);
          setHighlightAreas([]);
          return;
      }

      const pageRect = pageElement.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();
      const scaleFactorX = pageRect.width / pdfPageWidth;
      const scaleFactorY = pageRect.height / pdfPageHeight;

      if (isNaN(scaleFactorX) || isNaN(scaleFactorY) || scaleFactorX <= 0 || scaleFactorY <= 0) {
          console.error("HIGHLIGHT_DEBUG: Invalid scale factors calculated.", { scaleFactorX, scaleFactorY });
          setHighlightAreas([]);
          return;
      }

      const pdfLeft = bbox.l;
      const pdfRight = bbox.r;
      const pdfTop = bbox.t;
      const pdfBottom = bbox.b;
      const pdfWidth = pdfRight - pdfLeft;
      const pdfHeight = Math.abs(pdfTop - pdfBottom);

      let pdfTopFromTopOrigin;
      if (coordOrigin === 'BOTTOMLEFT') {
        pdfTopFromTopOrigin = pdfPageHeight - Math.max(pdfTop, pdfBottom);
      } else {
        pdfTopFromTopOrigin = Math.min(pdfTop, pdfBottom);
        // console.warn("HIGHLIGHT_DEBUG: Assuming TOPLEFT origin.");
      }

      let highlightLeft = pdfLeft * scaleFactorX;
      let highlightTop = pdfTopFromTopOrigin * scaleFactorY;
      let highlightWidth = pdfWidth * scaleFactorX;
      let highlightHeight = pdfHeight * scaleFactorY;

      highlightTop += (pageRect.top - containerRect.top);
      highlightLeft += (pageRect.left - containerRect.left);

      // Boundary checks remain the same
      const pageOffsetLeft = pageRect.left - containerRect.left;
      const pageOffsetTop = pageRect.top - containerRect.top;
       if (highlightLeft < pageOffsetLeft) {
           highlightWidth -= (pageOffsetLeft - highlightLeft);
           highlightLeft = pageOffsetLeft;
       }
       if (highlightLeft + highlightWidth > pageOffsetLeft + pageRect.width) {
           highlightWidth = pageOffsetLeft + pageRect.width - highlightLeft;
       }
       if (highlightTop < pageOffsetTop) {
           highlightHeight -= (pageOffsetTop - highlightTop);
           highlightTop = pageOffsetTop;
       }
       if (highlightTop + highlightHeight > pageOffsetTop + pageRect.height) {
           highlightHeight = pageOffsetTop + pageRect.height - highlightTop;
       }
       highlightWidth = Math.max(0, highlightWidth);
       highlightHeight = Math.max(0, highlightHeight);
       // --- End Boundary Checks ---

      const finalHighlight = { top: highlightTop, left: highlightLeft, width: highlightWidth, height: highlightHeight };
      console.log("HIGHLIGHT_DEBUG: Final Highlight Coords", finalHighlight);

      if (finalHighlight.width <= 0 || finalHighlight.height <= 0) {
          console.warn("HIGHLIGHT_DEBUG: Calculated highlight has zero or negative dimensions.", finalHighlight);
        setHighlightAreas([]);
      } else {
          console.log("HIGHLIGHT_DEBUG: Setting highlightAreas state.");
          setHighlightAreas([finalHighlight]);
          setTimeout(() => scrollToHighlight(finalHighlight), 0); 
      }

    } catch (error) {
      console.error("HIGHLIGHT_DEBUG: Error during transformation pipeline:", error);
      setHighlightAreas([]);
    }
  }, [currentChunk, pdfViewerRef, safePageNumber, pdfProxy, scrollToHighlight]); // Removed setters


  // --- RAG Query (askQuestion - UPDATED) ---
  const askQuestion = useCallback(async (question) => {
    if (!setName) { // Guard against missing setName
        setError("No document set selected.");
        return;
    }
    setIsLoading(true);
    setError('');
    const userMsgId = Date.now();
    setMessages(prev => [...prev, { id: userMsgId, sender: 'user', text: question }]);

    try {
      // UPDATED: Send set_name instead of knowledge_base_id
      const requestBody = { set_name: setName, question: question }; 
      const apiUrl = `${backendUrl}/rag`;
      console.log('Sending RAG request to:', apiUrl, 'Body:', JSON.stringify(requestBody));
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include', // Keep if needed for auth
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorDetail = errorText;
        try {
            const errorJson = JSON.parse(errorText);
            errorDetail = errorJson.detail || errorText; // Use detail if available
        } catch (parseError) {
             // Ignore if response is not JSON
        }
        throw new Error(`API request failed: ${response.status} ${response.statusText} - ${errorDetail}`);
      }

      const data = await response.json();
      console.log('Raw API response:', data);

      // Process and SORT chunks 
      const processedChunks = (data.chunks || [])
        .map((chunk, index) => {
          // Extract filename from the NEW URL format
          // Expecting url like: http://.../document_sets/SET_NAME/documents/FILENAME.pdf
          let fileName = 'Unknown Document';
          let chunkSetName = null;
          try {
             if (chunk.pdfUrl) {
                const url = new URL(chunk.pdfUrl); 
                const pathParts = url.pathname.split('/');
                // Find 'documents' segment, filename is after it
                const docIndex = pathParts.indexOf('documents');
                if (docIndex > -1 && docIndex < pathParts.length - 1) {
                   fileName = decodeURIComponent(pathParts[pathParts.length - 1]);
                   // Extract set name (assuming it's before 'documents')
                   if (docIndex > 0) {
                       chunkSetName = pathParts[docIndex - 1]; // Get the segment before 'documents'
                   }
                }
             }
          } catch (e) { console.error("Error parsing PDF URL:", chunk.pdfUrl, e); }

          const absolutePdfUrl = chunk.pdfUrl;
          const pageNum = chunk.position?.pageNumber || 1;
          const questionSim = chunk.metadata?.question_similarity ?? 0;
          const answerSim = chunk.metadata?.answer_similarity ?? 0;
          return {
            ...chunk,
            id: chunk.id || `${userMsgId + 1}-${index}`,
            key: `browser-chunk-${chunk.id || index}`,
            pdfUrl: absolutePdfUrl, 
            fileName: fileName,
            page: pageNum,
            questionSimilarity: questionSim,
            answerSimilarity: answerSim,
            // Store set name derived from URL for potential debug/display
            derivedSetName: chunkSetName, 
            // Properties for display in chunk browser (redundant but clear)
            displayFileName: fileName,
            displayPageNum: pageNum,
          };
        })
        // Sorting logic remains the same
        .sort((a, b) => {
            if (b.answerSimilarity !== a.answerSimilarity) {
                return b.answerSimilarity - a.answerSimilarity;
            }
            return b.questionSimilarity - a.questionSimilarity;
        });

      // --- FILTER chunks based on answer similarity --- (NEW)
      const filteredChunks = processedChunks.filter(
          chunk => chunk.answerSimilarity > 0.5
      );
      console.log(`Filtered ${processedChunks.length} raw chunks down to ${filteredChunks.length} with Answer Sim > 0.5`);

      const initialCount = processedChunks.length;
      const filteredCount = filteredChunks.length;
      let finalChunks = [];

      if (filteredCount >= 3) {
          // Enough high-similarity chunks, take up to 10
          finalChunks = filteredChunks.slice(0, 10);
          console.log(`Filtered count >= 3 (${filteredCount}). Taking top ${finalChunks.length} filtered chunks.`);
      } else if (initialCount >= 3) {
          // Not enough high-similarity chunks, but got at least 3 initially. Take top 3 overall.
          finalChunks = processedChunks.slice(0, 3);
          console.log(`Filtered count < 3 (${filteredCount}), but initial count >= 3 (${initialCount}). Taking top 3 overall chunks.`);
      } else {
          // Got less than 3 chunks initially. Show all of them.
          finalChunks = processedChunks;
          console.log(`Initial count < 3 (${initialCount}). Taking all ${finalChunks.length} initially retrieved chunks.`);
      }

      const botMsgId = userMsgId + 1;
      const botResponse = {
        id: botMsgId,
        sender: 'bot',
        text: data.answer || "I couldn't find an answer.",
        chunks: finalChunks // Use the final list here
      };

      console.log('Adding bot response (with final chunks logic applied):', botResponse);
      setMessages(prev => [...prev, botResponse]);

    } catch (error) {
      console.error('Error querying RAG API:', error);
      const errorMsgId = Date.now() + 1;
      const errorResponse = {
        id: errorMsgId,
        sender: 'bot',
        text: `Sorry, an error occurred: ${error.message}`,
        chunks: []
      };
      setMessages(prev => [...prev, errorResponse]);
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  // Dependencies: Added setName
  }, [setName, backendUrl, setMessages, setIsLoading, setError]);


  // --- Event Handlers (handleSubmit - unchanged) ---
  const handleSubmit = useCallback(/* ... unchanged ... */ (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      askQuestion(input);
      setInput('');
    }
  }, [input, isLoading, askQuestion]);

  // --- Event Handlers (handleChunkClick - Check URL logic) ---
  const handleChunkClick = useCallback((chunk) => {
    console.log("=== Handling Chunk Click ===", chunk);
    try {
        if (!chunk || !chunk.pdfUrl || !chunk.position) {
            console.error("Invalid chunk data for click:", chunk);
            setError("Cannot navigate: Invalid source data.");
            return;
        }

        const targetPage = chunk.position.pageNumber || 1;
        // Use the pdfUrl directly from the chunk data
        const newPdfOriginalUrl = chunk.pdfUrl; 

        console.log("Current Original URL State:", currentPdfOriginalUrl);
        console.log("New Clicked Original URL:", newPdfOriginalUrl);

        // Compare the original URLs. If different, or no PDF loaded, fetch.
        if (currentPdfOriginalUrl !== newPdfOriginalUrl || !currentPdf) {
            console.log("Loading new PDF required.");
            setHighlightAreas([]); // Clear highlights only when loading NEW PDF
            fetchPdf(newPdfOriginalUrl, targetPage, chunk); // Fetch using the URL from chunk
        } else {
            // Same PDF, just update page and context
            console.log("Same PDF, updating page and chunk context.");
            setPageNumber(targetPage);
            setCurrentChunk(chunk);
            // useEffect dependency on currentChunk will trigger transformCoordinates
        }
    } catch (error) {
      console.error('Error handling chunk click:', error);
        setError(`Failed to process source click: ${error.message}`);
    }
  // Dependencies: fetchPdf and state setters are implicitly included via useCallback rules
  // Explicit dependencies are state variables read directly for comparison.
  }, [currentPdfOriginalUrl, currentPdf, fetchPdf]); // Removed setters


  // --- PDF Callbacks (onDocumentLoadSuccess, onDocumentLoadError, onPageRenderSuccess, onPageLoadError - unchanged) ---
   const onDocumentLoadSuccess = useCallback(/* ... unchanged ... */ (pdfProxyFromLoad) => {
    console.log("%cHIGHLIGHT_DEBUG: === onDocumentLoadSuccess START === Received:", 'color: green; font-weight: bold;');
    // console.dir(pdfProxyFromLoad);
    const nextPdfProxy = pdfProxyFromLoad;
    const nextNumPages = nextPdfProxy?.numPages;

    if (!nextPdfProxy || typeof nextPdfProxy.getPage !== 'function') {
      console.error("HIGHLIGHT_DEBUG: onDocumentLoadSuccess received invalid PDFDocumentProxy!");
      setError("Internal error: Failed to get PDF document reference.");
      setPdfLoadingStatus('error');
      setPdfProxy(null); 
      return;
    }
    if (typeof nextNumPages !== 'number' || nextNumPages <= 0) {
      console.error("HIGHLIGHT_DEBUG: onDocumentLoadSuccess got invalid numPages from proxy!", nextNumPages);
      setError("Internal error: Invalid page count from PDF.");
      setPdfLoadingStatus('error');
      setPdfProxy(null); 
      return;
    }
    setNumPages(nextNumPages);
    setPdfProxy(nextPdfProxy);
    console.log("%cHIGHLIGHT_DEBUG: setPdfProxy STATE UPDATE CALLED with valid proxy object.", 'color: green; font-weight: bold;');
    setPdfLoadingStatus('success');
    console.log(`%cHIGHLIGHT_DEBUG: === onDocumentLoadSuccess END ===`, 'color: green; font-weight: bold;');
  }, [ ]); // No dependencies needed for setters

  const onDocumentLoadError = useCallback(/* ... unchanged ... */ (error) => {
      console.error('%cHIGHLIGHT_DEBUG: === onDocumentLoadError ===', 'color: red; font-weight: bold;', error);
      setError(`Failed to load PDF structure: ${error.message}`);
      setPdfLoadingStatus('error');
      setCurrentPdf(null);
      setCurrentPdfOriginalUrl(null);
      setNumPages(null);
      setPdfProxy(null);
      setCurrentChunk(null);
      setHighlightAreas([]);
      console.error('%cHIGHLIGHT_DEBUG: Cleared PDF state due to document load error.', 'color: red; font-weight: bold;');
  }, [ ]); // No dependencies needed for setters

  const onPageRenderSuccess = useCallback(/* ... unchanged ... */ () => {
    console.log(`HIGHLIGHT_DEBUG: onRenderSuccess finished for page ${safePageNumber}. Setting lastRenderedPage.`);
    setLastRenderedPage(safePageNumber);
  }, [safePageNumber]);

  const onPageLoadError = useCallback(/* ... unchanged ... */ (error) => {
      console.error(`Error loading/rendering page ${safePageNumber}:`, error);
      setError(`Failed to load page ${safePageNumber}: ${error.message}`);
  }, [safePageNumber, setError]);


  // --- Other Helpers (scrollToBottom, zoomIn, zoomOut, toggleDebugInfo - unchanged) ---
  const scrollToBottom = () => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); };
  const zoomIn = () => setScale(prev => Math.min(prev + 0.2, 3.0));
  const zoomOut = () => setScale(prev => Math.max(prev - 0.2, 0.5));
  const toggleDebugInfo = () => setShowDebugInfo(prev => !prev);

  // --- Toggle Chunk Browser (UPDATED for new chunk data structure if needed) ---
  const toggleChunkBrowser = useCallback(() => {
    if (!showChunkBrowser) {
        const lastBotMessageWithChunks = [...messages].reverse().find(
            msg => msg.sender === 'bot' && msg.chunks && msg.chunks.length > 0
        );
        
        if (lastBotMessageWithChunks) {
            console.log("Showing chunks from last bot message:", lastBotMessageWithChunks.chunks);
            // Use the pre-processed chunk data directly (it has displayFileName/PageNum)
            const chunksToDisplay = lastBotMessageWithChunks.chunks.map((chunk, index) => ({
                ...chunk,
                key: chunk.key || `browser-chunk-${chunk.id || index}`, // Ensure key
                // Ensure display properties exist, using fallbacks if necessary
                displayFileName: chunk.displayFileName || chunk.fileName || 'Unknown File',
                displayPageNum: chunk.displayPageNum || chunk.page || chunk.position?.pageNumber || 'N/A'
            }));
            setAllChunks(chunksToDisplay);
            setError(''); // Clear errors when showing chunks
        } else {
            console.log("No previous bot message with sources found.");
            setAllChunks([]);
            setError("No sources available to browse for the last answer."); // Set error
        }
    }
    setShowChunkBrowser(prev => !prev);
  }, [showChunkBrowser, messages, setError]); // Removed setters

  // ==================================
  // 5. EFFECTS (Unchanged, except for cleanup key)
  // ==================================
  useEffect(scrollToBottom, [messages]);

  // Effect to cleanup blob URL (Keyed by original URL now)
  useEffect(() => {
    const pdfToClean = currentPdf; 
    const urlKey = currentPdfOriginalUrl; // Use original URL as part of the key
    return () => {
      if (pdfToClean && pdfToClean.startsWith('blob:')) {
        console.log(`Cleanup Effect for ${urlKey}: Revoking blob URL:`, pdfToClean);
        URL.revokeObjectURL(pdfToClean);
      }
    };
  // Run cleanup when the original URL changes or component unmounts
  }, [currentPdf, currentPdfOriginalUrl]); 

  // useEffect for coordinate transformation (unchanged)
  useEffect(() => {
      console.log("HIGHLIGHT_DEBUG: Effect check - Transform Trigger/Consistency...",
          { hasProxy: !!pdfProxy, hasChunk: !!currentChunk, targetPage: currentChunk?.position?.pageNumber, lastRenderedPage });

      if (pdfProxy && currentChunk && currentChunk.position && lastRenderedPage === currentChunk.position.pageNumber) {
          console.log("%cHIGHLIGHT_DEBUG: Conditions met in Effect. Calling transformCoordinates.", 'color: blue; font-weight: bold;');
          transformCoordinates(); 
      } else {
          const reason = !pdfProxy ? "proxy missing"
                       : !currentChunk ? "chunk missing"
                       : !currentChunk.position ? "chunk pos missing"
                       : `last rendered page ${lastRenderedPage} !== chunk page ${currentChunk?.position?.pageNumber}`;
          console.log(`HIGHLIGHT_DEBUG: Effect clearing highlights because conditions not met (${reason}).`);
          setHighlightAreas([]);
      }
  }, [pdfProxy, currentChunk, lastRenderedPage, scale, transformCoordinates]); // Added setHighlightAreas dep if missing


  // ==================================
  // 6. JSX RETURN
  // ==================================
  return (
    <div className="app-container">
      {/* Chunk Browser Modal (Logic inside mostly unchanged, check props) */} 
      {showChunkBrowser && (
           <div className="chunk-browser-modal">
             <div className="chunk-browser-content">
               <div className="chunk-browser-header">
                 <h2>Sources for Last Answer ({allChunks.length} Chunks)</h2>
                 <button onClick={toggleChunkBrowser} className="chunk-browser-close-btn">
                   &times;
              </button>
            </div>
               <div className="chunk-browser-body">
                 {allChunks.length === 0 ? (
                     <div className="chunk-browser-empty">{error || "No sources found for the last answer."}</div> 
                 ) : (
                     <ul className="chunk-browser-list">
                       {allChunks.map((chunk) => (
                         <li key={chunk.key} className="chunk-browser-item">
                           <div className="chunk-item-header">
                <div>
                               <strong>File:</strong> {chunk.displayFileName} | 
                               <strong>Page:</strong> {chunk.displayPageNum}
                        </div>
                        <button
                               onClick={() => {
                                 handleChunkClick(chunk); 
                                 toggleChunkBrowser(); // Close modal
                               }}
                               className="view-chunk-btn"
                        >
                          View in PDF
                        </button>
                      </div>
                           <div className="chunk-item-text">
                             <pre>{chunk.text}</pre>
                      </div>
                           {/* Display metadata including similarity scores */} 
                           <details className="chunk-item-metadata">
                              <summary>
                                 View Details (Q: {chunk.metadata?.question_similarity?.toFixed(4) ?? 'N/A'}, 
                                 A: {chunk.metadata?.answer_similarity?.toFixed(4) ?? 'N/A'})
                        </summary>
                              <pre>{JSON.stringify({ 
                                  filename: chunk.metadata?.filename,
                                  page_no: chunk.metadata?.page_no,
                                  coord_origin: chunk.metadata?.coord_origin,
                                  bbox_json: chunk.metadata?.bbox_json,
                                  question_similarity: chunk.metadata?.question_similarity,
                                  answer_similarity: chunk.metadata?.answer_similarity
                               }, null, 2)}</pre>
                      </details>
                         </li>
                       ))}
                     </ul>
              )}
            </div>
               <div className="chunk-browser-footer">
                 <button onClick={toggleChunkBrowser}>Close</button>
            </div>
          </div>
        </div>
      )}
      
      {/* Main Chat and PDF Layout */} 
      <div className="chat-container">
        <div className="chat-header">
           {/* Use the onBack prop passed from App */} 
           <button className="back-button" onClick={onBack}>← Back to Sets</button>
           {/* Header Title: Display current set name */} 
           <div className="chat-title">Set: {setName}</div>
           <div className="header-buttons">
              {/* <button className="chunk-browser-toggle" onClick={toggleChunkBrowser}>
                  Browse Sources
              </button> */}
              {/* Debug button can remain if needed */} 
              {/* <button className="debug-button" onClick={toggleDebugInfo}> 
                  {showDebugInfo ? 'Hide Debug' : 'Show Debug'}
              </button> */} 
           </div>
        </div>
        <div className="chat-messages">
          {messages.length === 0 && (
            <div className="welcome-message">
                    <h3>Chat with: {setName}</h3>
                    <p>Ask a question about the documents in this set.</p>
            </div>
          )}
          {messages.map((message) => (
            <div key={message.id} className={`message ${message.sender}`}>
              <div className="message-content">{message.text}</div>
                 {/* Source/Chunk Display */} 
                 {message.sender === 'bot' && message.chunks && message.chunks.length > 0 && (
                <div className="chunks">
                     <div className="chunks-header">Sources (Top {message.chunks.length}):</div>
                  {message.chunks.map((chunk, index) => (
                    <div 
                         key={chunk.key || `chunk-${message.id}-${index}`}
                      className="chunk"
                      onClick={() => handleChunkClick(chunk)}
                         title={`Q: ${chunk.metadata?.question_similarity?.toFixed(4)}, A: ${chunk.metadata?.answer_similarity?.toFixed(4)}`}
                    >
                      <div className="chunk-text">{chunk.text.substring(0, 150)}...</div>
                      <div className="chunk-source">
                           Source {index + 1}: {chunk.fileName} (p. {chunk.page})
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
            {isLoading && <div className="loading chat-loading">Thinking...</div>}
          <div ref={messagesEndRef} />
        </div>
        <form className="chat-input" onSubmit={handleSubmit}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !input.trim()}>Send</button>
        </form>
      </div>
      
      {/* PDF Viewer Area */} 
      <div className="pdf-container">
        {pdfLoadingStatus === 'loading' && <div className="pdf-loading"><p>Loading PDF...</p></div>}
        {pdfLoadingStatus === 'error' && <div className="no-pdf error-message">Error: {error}</div>}

        {currentPdf && pdfLoadingStatus !== 'error' ? (
          <>
            <div className="pdf-controls">
               <button disabled={safePageNumber <= 1} onClick={() => setPageNumber(prev => Math.max(1, prev - 1))}>Previous</button>
               <span>Page {safePageNumber} of {numPages || '...'}</span>
               <button disabled={safePageNumber >= (numPages || 1)} onClick={() => setPageNumber(prev => Math.min(numPages || 1, prev + 1))}>Next</button>
              <div className="zoom-controls">
                <button onClick={zoomOut}>-</button>
                <span>{Math.round(scale * 100)}%</span>
                <button onClick={zoomIn}>+</button>
              </div>
            </div>
            <div className="pdf-viewer" ref={pdfViewerRef}>
              <Document
                file={currentPdf}
                onLoadSuccess={onDocumentLoadSuccess}
                onLoadError={onDocumentLoadError}
                loading={<div className="pdf-loading-indicator">Loading Document Structure...</div>}
                options={optionsRef.current}
              >
                {numPages && numPages > 0 ? (
                  <Page 
                    key={`page_${safePageNumber}_${scale}`}
                    pageNumber={safePageNumber}
                    scale={scale}
                    onRenderSuccess={onPageRenderSuccess} // Use the memoized callback
                    onLoadError={onPageLoadError} // Use the memoized callback
                    loading={<div className="page-loading-indicator">Rendering Page {safePageNumber}...</div>}
                    className="pdf-page-rendered"
                  />
                ) : pdfLoadingStatus === 'success' ? (
                    <div className="pdf-loading-indicator">Waiting for page info...</div>
                ) : null }
              </Document>
              
              {/* Dynamic highlights (unchanged logic) */} 
              {highlightAreas.map((highlight, index) => {
                 return highlight && highlight.width > 0 && highlight.height > 0 ? (
                  <div 
                    key={`highlight-${index}`}
                    className="highlight-overlay"
                    style={{
                      position: 'absolute',
                      top: `${highlight.top}px`,
                      left: `${highlight.left}px`,
                      width: `${highlight.width}px`,
                      height: `${highlight.height}px`,
                       backgroundColor: 'rgba(255, 255, 0, 0.5)',
                      pointerEvents: 'none',
                      zIndex: 10,
                       mixBlendMode: 'multiply'
                    }}
                  />
                 ) : null;
              })}
            </div>
          </>
        ) : pdfLoadingStatus !== 'loading' && (
          <div className="no-pdf">
            <h3>PDF Viewer</h3>
            <p>Select a document set and ask a question, then click a source to view.</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;