class ModelCache {
    private readonly dbName: string;
    private readonly dbVersion: number;
    private readonly storeName: string;

    constructor(dbName: string = 'modelDB', dbVersion: number = 1, storeName: string = 'models') {
        this.dbName = dbName;
        this.dbVersion = dbVersion;
        this.storeName = storeName;
    }

    private async initDB(): Promise<IDBDatabase> {
        return new Promise<IDBDatabase>((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.dbVersion);

            request.onupgradeneeded = (event: IDBVersionChangeEvent) => {
                const db = (event.target as any).result as IDBDatabase;
                if (!db.objectStoreNames.contains(this.storeName)) {
                    db.createObjectStore(this.storeName);
                }
            };

            request.onsuccess = (event: Event) => {
                const db = (event.target as any).result as IDBDatabase;
                resolve(db);
            };

            request.onerror = (event: Event) => {
                reject((event.target as any).error);
            };
        });
    }

    async saveModel(url: string, modelName: string, onProgress: (progress: number) => void): Promise<void> {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error('Failed to fetch model');
        }

        const contentLength = response.headers.get('content-length');
        if (!contentLength) {
            throw new Error('Content-Length header is missing');
        }

        const totalBytes = parseInt(contentLength, 10);
        let loadedBytes = 0;

        const reader = response.body?.getReader();
        if (!reader) {
            throw new Error('Failed to get reader from response body');
        }

        const stream = new ReadableStream({
            async start(controller) {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) {
                        break;
                    }
                    loadedBytes += value.byteLength;
                    onProgress((loadedBytes / totalBytes) * 100);
                    controller.enqueue(value);
                }
                controller.close();
                reader.releaseLock();
            }
        });

        const arrayBuffer = await new Response(stream).arrayBuffer();

        const db = await this.initDB();
        const transaction = db.transaction([this.storeName], 'readwrite');
        const store = transaction.objectStore(this.storeName);
        const request = store.put(arrayBuffer, modelName);

        return new Promise<void>((resolve, reject) => {
            request.onsuccess = () => {
                resolve();
            };
            request.onerror = (event) => {
                reject((event.target as any).error);
            };
        });
    }

    async getModel(modelName: string): Promise<ArrayBuffer> {
        const db = await this.initDB();
        const transaction = db.transaction([this.storeName], 'readonly');
        const store = transaction.objectStore(this.storeName);
        const request = store.get(modelName);

        return new Promise<ArrayBuffer>((resolve, reject) => {
            request.onsuccess = (event) => {
                const result = (event.target as any).result as ArrayBuffer;
                if (result) {
                    resolve(result);
                } else {
                    reject(new Error('Model not found'));
                }
            };
            request.onerror = (event) => {
                reject((event.target as any).error);
            };
        });
    }

    async loadModel(url: string, modelName: string, onProgress: (progress: number) => void): Promise<ArrayBuffer> {
        try {
            const model = await this.getModel(modelName);
            console.log('Model loaded from IndexedDB');
            return model;
        } catch (error) {
            console.log('Model not found in IndexedDB, fetching from network');
            await this.saveModel(url, modelName, onProgress);
            const model = await this.getModel(modelName);
            return model;
        }
    }
}

export default ModelCache;
