/// <reference types="vite/client" />
interface ImportMetaEnv { readonly VITE_WORKER_BASE?: string; }
interface ImportMeta { readonly env: ImportMetaEnv; }
