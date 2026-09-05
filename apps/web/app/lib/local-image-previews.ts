export type LocalMessageImage = {
  mime_type: string;
  data: string;
  preview_url?: string;
  name?: string;
};

type StoredMessageImages = {
  key: string;
  conversationId: string;
  messageId: string;
  images: Omit<LocalMessageImage, "preview_url">[];
  updatedAt: string;
};

const DB_NAME = "archimedes-local-message-images";
const DB_VERSION = 1;
const STORE_NAME = "message_images";

function canUseIndexedDb() {
  return typeof window !== "undefined" && "indexedDB" in window;
}

function storageKey(conversationId: string, messageId: string) {
  return `${conversationId}:${messageId}`;
}

function openImageDb(): Promise<IDBDatabase | null> {
  if (!canUseIndexedDb()) return Promise.resolve(null);

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "key" });
        store.createIndex("conversationId", "conversationId", { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Could not open image preview storage"));
  });
}

function withPreviewUrl(image: Omit<LocalMessageImage, "preview_url">): LocalMessageImage {
  return {
    ...image,
    preview_url: `data:${image.mime_type};base64,${image.data}`,
  };
}

export async function saveMessageImages(
  conversationId: string,
  messageId: string,
  images: LocalMessageImage[],
) {
  try {
    if (images.length === 0) return;
    const db = await openImageDb();
    if (!db) return;

    const storedImages = images.map(({ mime_type, data, name }) => ({ mime_type, data, name }));
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      tx.objectStore(STORE_NAME).put({
        key: storageKey(conversationId, messageId),
        conversationId,
        messageId,
        images: storedImages,
        updatedAt: new Date().toISOString(),
      } satisfies StoredMessageImages);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error("Could not save image previews"));
    });
    db.close();
  } catch (error) {
    console.warn("Could not save local image previews", error);
  }
}

export async function loadMessageImages(
  conversationId: string,
  messageId: string,
): Promise<LocalMessageImage[]> {
  try {
    const db = await openImageDb();
    if (!db) return [];

    const record = await new Promise<StoredMessageImages | undefined>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const request = tx.objectStore(STORE_NAME).get(storageKey(conversationId, messageId));
      request.onsuccess = () => resolve(request.result as StoredMessageImages | undefined);
      request.onerror = () => reject(request.error ?? new Error("Could not load image previews"));
    });
    db.close();

    return (record?.images ?? []).map(withPreviewUrl);
  } catch (error) {
    console.warn("Could not load local image previews", error);
    return [];
  }
}

export async function deleteConversationImages(conversationId: string) {
  try {
    const db = await openImageDb();
    if (!db) return;

    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const index = tx.objectStore(STORE_NAME).index("conversationId");
      const request = index.openCursor(IDBKeyRange.only(conversationId));

      request.onsuccess = () => {
        const cursor = request.result;
        if (!cursor) return;
        cursor.delete();
        cursor.continue();
      };
      request.onerror = () => reject(request.error ?? new Error("Could not delete image previews"));
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error("Could not delete image previews"));
    });
    db.close();
  } catch (error) {
    console.warn("Could not delete local image previews", error);
  }
}
