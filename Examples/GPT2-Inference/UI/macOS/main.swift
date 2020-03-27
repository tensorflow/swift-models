// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import AppKit

import Dispatch
import Foundation

import TextModels

internal class NSLabel: NSTextField {
  public override init(frame: NSRect) {
    super.init(frame: frame)

    self.drawsBackground = false
    self.isBezeled = false
    self.isEditable = false
    self.isSelectable = false
  }

  public required init?(coder: NSCoder) {
    fatalError("init?(coder:) not implemented")
  }
}

internal extension NSLabel {
  convenience init(frame: NSRect, title: String) {
    self.init(frame: frame)
    self.stringValue = title
  }
}

internal extension NSButton {
  convenience init(frame: NSRect, title: String) {
    self.init(frame: frame)
    self.title = title
  }

  var text: String {
    get { return self.stringValue }
    set { self.stringValue = newValue }
  }
}

internal extension NSTextField {
  var text: String {
    get { return self.stringValue }
    set { self.stringValue = newValue }
  }
}

internal extension NSTextView {
  var text: String {
    get { return self.string }
    set { self.string = newValue }
  }
}

internal extension NSSlider {
  var value: Double {
    get { return self.doubleValue }
    set { self.doubleValue = newValue }
  }
}

func onMain(_ body: @escaping () -> ()) {
  if Thread.isMainThread {
    body()
  } else {
    DispatchQueue.main.async { body() }
  }
}

func onBackground(_ body: @escaping () -> ()) {
  DispatchQueue.global(qos: .background).async { body() }
}

class SwiftApplicationDelegate: NSObject, NSApplicationDelegate {
  var window: NSWindow =
      NSWindow(contentRect: NSRect(x: 0, y: 0, width: 648, height: 432),
               styleMask: [.titled, .closable, .miniaturizable, .resizable],
               backing: .buffered,
               defer: false)

  lazy var input: NSTextField =
      NSTextField(frame: NSRect(x: 24, y: 392, width: 512, height: 20))
  lazy var button: NSButton =
      NSButton(frame: NSRect(x: 544, y: 386, width: 72, height: 32),
               title: "Generate")
  lazy var output: NSTextView =
      NSTextView(frame: NSRect(x: 24, y: 128, width: 512, height: 256))
  lazy var slider: NSSlider =
      NSSlider(frame: NSRect(x: 24, y: 100, width: 512, height: 32))
  lazy var label: NSLabel =
      NSLabel(frame: NSRect(x: 24, y: 60, width: 512, height: 20),
              title: "Loading GPT-2 ...")

  var gpt: GPT2?

  func applicationDidFinishLaunching(_ notification: Notification) {
    self.slider.minValue = 0.0
    self.slider.maxValue = 1.0
    self.slider.value = 0.5

    self.window.contentView?.addSubview(self.input)
    self.window.contentView?.addSubview(self.button)

    let view: NSScrollView = NSScrollView(frame: self.output.frame)
    view.hasVerticalScroller = true
    view.hasHorizontalScroller = true
    view.documentView = self.output
    self.output.textContainer?.widthTracksTextView = false
    self.window.contentView?.addSubview(view)

    // self.window.contentView?.addSubview(self.output)
    self.window.contentView?.addSubview(self.slider)
    self.window.contentView?.addSubview(self.label)

    let ComicSansMS: NSFont = NSFont(name: "Comic Sans MS", size: 10)!

    self.input.font = ComicSansMS
    self.input.stringValue = "Introducing Swift for TensorFlow on macOS"

    self.label.font = ComicSansMS
    self.output.font = ComicSansMS
    self.output.isEditable = false

    self.button.target = self
    self.button.action = #selector(generate)

    onBackground {
      do {
        self.gpt = try GPT2()
        onMain { self.label.text = "GPT-2 ready!" }
      } catch {
        onMain { self.label.text = "GPT-2 Construction Failure" }
      }
    }

    self.window.makeKeyAndOrderFront(nil)
  }

  func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication)
      -> Bool {
    return true
  }
}

extension SwiftApplicationDelegate {
  @objc
  func generate() {
    guard let gpt = self.gpt else { return }

    output.text = input.text
    if !input.text.isEmpty {
      gpt.seed = gpt.embedding(for: input.text)
    }
    gpt.temperature = Float(self.slider.value)

    onBackground {
      for _ in 0 ..< 256 {
        do {
          let word: String = try gpt.generate()
          onMain {
            self.output.text = self.output.text + word
            let range: NSRange =
                NSRange(location: self.output.text.count - 1, length: 1)
            self.output.scrollRangeToVisible(range)
          }
        } catch {
          continue
        }
      }
    }
  }
}

let delegate: SwiftApplicationDelegate = SwiftApplicationDelegate()
NSApplication.shared.delegate = delegate
_ = NSApplicationMain(CommandLine.argc, CommandLine.unsafeArgv)
