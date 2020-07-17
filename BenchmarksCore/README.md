## Adding new settings 

Let's walk through the process of adding a new String-typed setting:

1. Add a new setting to [BenchmarkSettings.swift](https://github.com/tensorflow/swift-models/blob/master/BenchmarksCore/BenchmarkSettings.swift):

    a. Add a new setting type:

    ```swift

    public struct MySetting: BenchmarkSetting {
      var value: String
      init(_ value: String) {
        self.value = value
      }
    }
    ```

    b. Add a default (if appropriate) to the `defaultSettings` in the same file:

    ```swift
    public let defaultSettings: [BenchmarkSetting] = [
      ...
      MySetting("...") // default value goes here
      ...
    ]
    ```

    c. Add a convenience getter to the `BenchmarkSettings` extension:

    ```swift
      var mySetting: String? {
        return self[MySetting.self]?.value
      }
    ```

    If your setting has a default you can unwrap the optional and return non-optional result:

    ```swift
      var mySetting: String {
        if let value = self[MySetting.self]?.value {
          return value
        } else {
          fatalError("MySetting setting must have a default.")
        }
      }
    ```

4. Add a new benchmark flag to [BenchmarkArguments.swift](https://github.com/tensorflow/swift-models/blob/master/BenchmarksCore/BenchmarkArguments.swift):

    a. Add a flag property:

    ```swift
    @Option(name: .customLong("mySettingFlagName"), help: "Useful description of MySetting here.")
    var mySettingFlagName: String?
    ```

    b. Validate your flag value in `BenchmarkArguments.validate()` function if necessary. 

    c. Convert command-line flag to a setting in `BenchmarkArgument.settings` computed property:

    ```swift
      if let value = mySettingFlagName {
        settings.append(MySetting(value))
      }
    ```
