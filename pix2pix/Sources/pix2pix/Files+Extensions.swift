import Files

extension Folder {
    func files(extensions: Set<String>) -> [File] {
        return self.files.filter { file in
            guard let ext = file.extension
            else { return false }

            return extensions.contains(ext)
        }
    }
}
