import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default function App() {
  const inputRef = useRef(null);

  /* =========================================================
     STATE
     ========================================================= */

  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("chatrieval-theme") || "dark";
  });

  const [files, setFiles] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [skipped, setSkipped] = useState([]);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");

  const [processing, setProcessing] = useState(false);
  const [asking, setAsking] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");


  /* =========================================================
     THEME
     ========================================================= */

  useEffect(() => {
    localStorage.setItem("chatrieval-theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((currentTheme) =>
      currentTheme === "dark" ? "light" : "dark"
    );
  };


  /* =========================================================
     FILE SELECTION
     ========================================================= */

  const selectFiles = (selected) => {
  const selectedFiles = Array.from(selected).filter(
    (file) =>
      file.type === "application/pdf" ||
      file.name.toLowerCase().endsWith(".pdf")
  );

  if (!selectedFiles.length) {
    setError("Please select PDF files only.");
    return;
  }

  setFiles((currentFiles) => {
    const existingKeys = new Set(
      currentFiles.map(
        (file) => `${file.name}-${file.size}-${file.lastModified}`
      )
    );

    const newFiles = selectedFiles.filter(
      (file) =>
        !existingKeys.has(
          `${file.name}-${file.size}-${file.lastModified}`
        )
    );

    return [...currentFiles, ...newFiles];
  });

  setError("");
};


  /* =========================================================
     UPLOAD AND PROCESS DOCUMENTS
     ========================================================= */

  const uploadFiles = async () => {
    if (!files.length) {
      setError("Select at least one PDF.");
      return;
    }

    setProcessing(true);
    setError("");

    const body = new FormData();

    files.forEach((file) => {
      body.append("files", file);
    });

    try {
      const response = await fetch(
        `${API_URL}/api/documents/upload`,
        {
          method: "POST",
          body,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Document processing failed."
        );
      }

      setSessionId(data.session_id);
      setDocuments(data.documents || []);
      setSkipped(data.skipped_pages || []);
      setMessages([]);
    } catch (err) {
      setError(
        err.message || "Document processing failed."
      );
    } finally {
      setProcessing(false);
    }
  };


  /* =========================================================
     ASK QUESTION
     ========================================================= */

  const ask = async (event) => {
    event?.preventDefault();

    const q = question.trim();

    if (!q) {
      return;
    }

    if (!sessionId) {
      setError(
        "Upload and process your PDFs first."
      );
      return;
    }

    setAsking(true);
    setError("");

    /* Add user message immediately */
    setMessages((currentMessages) => [
      ...currentMessages,
      {
        role: "user",
        content: q,
      },
    ]);

    setQuestion("");

    try {
      const response = await fetch(
        `${API_URL}/api/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            session_id: sessionId,
            question: q,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Failed to generate answer."
        );
      }

      /* Add assistant response */
      setMessages((currentMessages) => [
        ...currentMessages,
        {
          role: "assistant",
          content: data.answer || "",
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      setError(
        err.message || "Failed to generate answer."
      );
    } finally {
      setAsking(false);
    }
  };


  /* =========================================================
     RESET / NEW CONVERSATION
     ========================================================= */

  const reset = () => {
    setSessionId(null);
    setFiles([]);
    setDocuments([]);
    setSkipped([]);
    setMessages([]);
    setQuestion("");
    setError("");
  };


  /* =========================================================
     RENDER
     ========================================================= */

  return (
    <div className={`app-shell ${theme}`}>

      {/* =====================================================
          SIDEBAR
          ===================================================== */}

      <aside className="sidebar">

        {/* Brand */}
        <div className="brand">

          <div className="brand-mark">
            C
          </div>

          <div>
            <div className="brand-name">
              Chatrieval
            </div>

            <div className="brand-subtitle">
              Document Intelligence
            </div>
          </div>

        </div>


        {/* New conversation */}
        <button
          className="new-chat"
          onClick={reset}
          type="button"
        >
          + New conversation
        </button>


        {/* =================================================
            DOCUMENT UPLOAD
            ================================================= */}

        <div className="sidebar-section">

          <div className="section-label">
            DOCUMENTS
          </div>


          {/* Upload / Drop Zone */}
          <div
  className={`drop-zone ${dragging ? "dragging" : ""}`}
  onClick={() => inputRef.current?.click()}
  onDragOver={(event) => {
    event.preventDefault();
    setDragging(true);
  }}
  onDragLeave={() => {
    setDragging(false);
  }}
  onDrop={(event) => {
    event.preventDefault();
    setDragging(false);
    selectFiles(event.dataTransfer.files);
  }}
>
  <input
    ref={inputRef}
    type="file"
    accept=".pdf,application/pdf"
    multiple
    hidden
    onChange={(event) => {
      selectFiles(event.target.files);

      // Allows selecting the same file again later
      event.target.value = "";
    }}
  />

  <div className="upload-icon">
    ↑
  </div>

  <strong>
    Drop multiple PDFs here
  </strong>

  <span>
    or click to browse
  </span>

  <small className="upload-hint">
    You can select multiple documents
  </small>
</div>


          {/* Selected files */}
          {files.length > 0 && (
  <div className="selected-files">

    <div className="selected-files-header">
      <span>
        Selected documents
      </span>

      <strong>
        {files.length}
      </strong>
    </div>

    {files.map((file, index) => (
      <div
        className="file-item"
        key={`${file.name}-${file.size}-${file.lastModified}`}
      >
        <div className="file-icon">
          PDF
        </div>

        <div className="file-info">
          <span title={file.name}>
            {file.name}
          </span>

          <small>
            {(file.size / 1024 / 1024).toFixed(2)} MB
          </small>
        </div>

        <button
          type="button"
          className="remove-file"
          onClick={() => {
            setFiles((currentFiles) =>
              currentFiles.filter(
                (_, fileIndex) =>
                  fileIndex !== index
              )
            );
          }}
          title="Remove PDF"
        >
          ×
        </button>
      </div>
    ))}

  </div>
)}


          {/* Process button */}
          <button
            className="process-button"
            disabled={
              !files.length ||
              processing
            }
            onClick={uploadFiles}
            type="button"
          >
            {processing
              ? "Processing..."
              : "Process documents"}
          </button>

        </div>


        {/* =================================================
            INDEX STATUS
            ================================================= */}

        {documents.length > 0 && (
          <div className="sidebar-section">

            <div className="section-label">
              INDEX STATUS
            </div>

            <div className="status-card">

              <div className="status-dot" />

              <div>
                <strong>
                  Ready
                </strong>

                <span>
                  {documents.length} document(s)
                  indexed
                </span>
              </div>

            </div>


            {/* Skipped image-only pages */}
            {skipped.length > 0 && (
              <p className="muted-note">
                {skipped.length} image-only
                page(s) skipped. OCR is
                intentionally disabled.
              </p>
            )}

          </div>
        )}


        {/* Sidebar technology footer */}
        <div className="sidebar-footer">

          <span>
            Local embeddings
          </span>

          <span>
            •
          </span>

          <span>
            FAISS
          </span>

          <span>
            •
          </span>

          <span>
            Groq
          </span>

        </div>

      </aside>


      {/* =====================================================
          MAIN PANEL
          ===================================================== */}

      <main className="main-panel">

        {/* =================================================
            TOP BAR
            ================================================= */}

        <header className="topbar">

          <div>

            <span className="eyebrow">
              AI DOCUMENT ASSISTANT
            </span>

            <h1>
              Ask your documents.
            </h1>

          </div>


          <div className="topbar-actions">

            {/* System status */}
            <div className="system-status">

              <span className="status-dot" />

              System online

            </div>


            {/* Theme toggle */}
            <button
              className="theme-toggle"
              onClick={toggleTheme}
              type="button"
              aria-label={`Switch to ${
                theme === "dark"
                  ? "light"
                  : "dark"
              } mode`}
              title={`Switch to ${
                theme === "dark"
                  ? "light"
                  : "dark"
              } mode`}
            >

              {theme === "dark" ? (
                <>
                  <span className="theme-icon">
                    ☀
                  </span>

                  <span>
                    Light
                  </span>
                </>
              ) : (
                <>
                  <span className="theme-icon">
                    ☾
                  </span>

                  <span>
                    Dark
                  </span>
                </>
              )}

            </button>

          </div>

        </header>


        {/* =================================================
            WELCOME SCREEN
            ================================================= */}

        {messages.length === 0 ? (

          <section className="welcome">

            

            <h2>
              Turn your PDFs into
              <span> answers.</span>
            </h2>

            <p>
              Upload your documents, build a
              local semantic index, and ask
              questions with source-aware
              answers.
            </p>


            {/* Feature cards */}
            <div className="feature-grid">

              <div className="feature-card">

                <div className="feature-number">
                  01
                </div>

                <h3>
                  Upload
                </h3>

                <p>
                  Drop one or multiple PDF
                  documents into the workspace.
                </p>

              </div>


              <div className="feature-card">

                <div className="feature-number">
                  02
                </div>

                <h3>
                  Retrieve
                </h3>

                <p>
                  Find information.
                </p>

              </div>


              <div className="feature-card">

                <div className="feature-number">
                  03
                </div>

                <h3>
                  Ask
                </h3>

                <p>
                  Ask Queries.
                </p>

              </div>

            </div>


            <div className="empty-hint">
              Start by uploading a PDF from
              the left panel.
            </div>

          </section>

        ) : (

          /* =================================================
             CHAT AREA
             ================================================= */

          <section className="chat-area">

            {messages.map(
              (message, index) => (

                <div
                  className={`message-row ${message.role}`}
                  key={`${message.role}-${index}`}
                >

                  {/* Avatar */}
                  <div className="avatar">
                    {message.role === "user"
                      ? "U"
                      : "C"}
                  </div>


                  <div className="message-content">

                    {/* Message label */}
                    <div className="message-label">
                      {message.role === "user"
                        ? "You"
                        : "Chatrieval"}
                    </div>


                    {/* Message */}
                    <div className="message-text">

                      {message.role === "assistant" ? (

                        <ReactMarkdown
                          remarkPlugins={[
                            remarkGfm,
                          ]}
                          skipHtml={true}
                        >
                          {String(
                            message.content || ""
                          )}
                        </ReactMarkdown>

                      ) : (

                        <span>
                          {message.content}
                        </span>

                      )}

                    </div>


                    {/* =================================================
                        SOURCES
                        ================================================= */}

                    {message.sources?.length > 0 && (

                      <div className="sources">

                        <div className="sources-title">
                          Sources
                        </div>


                        {message.sources.map(
                          (source, sourceIndex) => (

                            <div
                              className="source-chip"
                              key={`${source.source}-${source.page}-${sourceIndex}`}
                            >

                              <span>
                                {source.source}
                              </span>

                              <b>
                                p. {source.page}
                              </b>

                            </div>

                          )
                        )}

                      </div>

                    )}

                  </div>

                </div>

              )
            )}


            {/* =================================================
                TYPING INDICATOR
                ================================================= */}

            {asking && (

              <div className="message-row assistant">

                <div className="avatar">
                  C
                </div>

                <div className="message-content">

                  <div className="message-label">
                    Chatrieval
                  </div>

                  <div className="typing">

                    <span />
                    <span />
                    <span />

                  </div>

                </div>

              </div>

            )}

          </section>

        )}


        {/* =================================================
            ERROR BANNER
            ================================================= */}

        {error && (

          <div className="error-banner">

            <strong>
              Something went wrong
            </strong>

            <span>
              {error}
            </span>

          </div>

        )}


        {/* =================================================
            CHAT COMPOSER
            ================================================= */}

        <form
          className="composer"
          onSubmit={ask}
        >

          <div className="composer-inner">

            <input
              value={question}
              onChange={(event) =>
                setQuestion(event.target.value)
              }
              placeholder={
                sessionId
                  ? "Ask anything about your documents..."
                  : "Start asking questions..."
              }
              disabled={
                !sessionId ||
                asking
              }
            />


            <button
              type="submit"
              disabled={
                !sessionId ||
                !question.trim() ||
                asking
              }
              aria-label="Send question"
            >
              ↑
            </button>

          </div>


          <div className="composer-note">
            Answers are grounded only in
            extracted PDF text.
          </div>

        </form>

      </main>

    </div>
  );
}