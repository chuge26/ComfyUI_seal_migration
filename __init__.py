from .nodes import PDFLoader, SealMigration, PDFSaver

NODE_CLASS_MAPPINGS = {
    "PDFLoader": PDFLoader,
    "SealMigration": SealMigration,
    "PDFSaver": PDFSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDFLoader": "📄 PDF加载器",
    "SealMigration": "🔄 PDF印章迁移器",
    "PDFSaver": "💾 PDF保存"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
