// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

/// A group whose stones are connected and share the same liberty.
struct LibertyGroup {
    // A numerical unique ID for the group.
    var id: Int

    var color: Color

    // The stones belonging to this group.
    var stones: Set<Position>

    // The liberties for this group.
    var liberties: Set<Position>
}

/// Tracks the liberty of all stones on board.
///
/// `LibertyTracker` is designed to be a struct as it trackes the liberty
/// information of current board snapshot. So not expected to be changed. After
/// placing a new stone, we make a copy, update it and then attach it to new
/// board snapshot to track state.
struct LibertyTracker {

    private let gameConfiguration: GameConfiguration

    // Tracks the liberty groups. For a position (stone) having no group,
    // groupIndex[stone] should be nil. Otherwise, the group ID should be
    // groupIndex[stone] and its group is groups[groupIndex[stone]].
    // The invariance check can be done via checkLibertyGroupsInvariance helper
    // method.
    private var nextGroupIDToAssign = 0
    private var groupIndex: [[Int?]]
    private var groups: [Int: LibertyGroup] = [:]

    init(gameConfiguration: GameConfiguration) {
        self.gameConfiguration = gameConfiguration

        let size = gameConfiguration.size
        groupIndex = Array(repeating: Array(repeating: nil, count: size), count: size)
    }

    /// Returns the liberty group at the position.
    func group(at position: Position) -> LibertyGroup? {
        guard let groupID = groupIndex(for: position) else {
            return nil
        }
        guard let group = groups[groupID] else {
            fatalErrorForGroupsInvariance(groupID: groupID)
        }
        return group
    }
}

/// Extend `LibertyTracker` to have a mutating method by placing a new stone.
extension LibertyTracker {

    /// Adds a new stone to the board and returns all captured stones.
    mutating func addStone(at position: Position, withColor color: Color) throws -> Set<Position> {
        precondition(groupIndex(for: position) == nil)

        printDebugInfo(message: "Before adding stone.")

        var capturedStones = Set<Position>()

        // Records neighbors information.
        var emptyNeighbors = Set<Position>()
        var opponentNeighboringGroupIDs = Set<Int>()
        var friendlyNeighboringGroupIDs = Set<Int>()

        for neighbor in position.neighbors(boardSize: gameConfiguration.size) {

            // First, handle the case neighbor has no group.
            guard let neighborGroupID = groupIndex(for: neighbor) else {
                emptyNeighbors.insert(neighbor)
                continue
            }

            guard let neighborGroup = groups[neighborGroupID] else {
                fatalErrorForGroupsInvariance(groupID: neighborGroupID)
            }

            if neighborGroup.color == color {
                friendlyNeighboringGroupIDs.insert(neighborGroupID)
            } else {
                opponentNeighboringGroupIDs.insert(neighborGroupID)
            }
        }

        if gameConfiguration.isVerboseDebuggingEnabled {
            print("empty: \(emptyNeighbors)")
            print("friends: \(friendlyNeighboringGroupIDs)")
            print("opponents: \(opponentNeighboringGroupIDs)")
        }

        // Creates new group and sets its liberty as the empty neighbors at first.
        var newGroupID = makeGroup(
            color: color,
            stone: position,
            liberties: emptyNeighbors
        ).id

        // Merging all friend groups.
        for friendGroupID in friendlyNeighboringGroupIDs {
            newGroupID = mergeGroups(newGroupID, friendGroupID)
        }

        // Calculates captured stones.
        for opponentGroupID in opponentNeighboringGroupIDs {
            guard var opponentGroup = groups[opponentGroupID] else {
                fatalErrorForGroupsInvariance(groupID: opponentGroupID)
            }

            guard opponentGroup.liberties.count > 1 else {
                // The only liberty will be taken by the stone just placed. Delete it.
                capturedStones.formUnion(captureGroup(opponentGroupID))
                continue
            }

            // Updates the liberty taken by the stone just placed.
            opponentGroup.liberties.remove(position)
            // As group is struct, we need to explicitly write it back.
            groups[opponentGroupID] = opponentGroup
            assert(checkLibertyGroupsInvariance())
        }

        if gameConfiguration.isVerboseDebuggingEnabled {
            print("captured stones: \(capturedStones)")
        }

        // Update liberties for existing stones
        updateLibertiesAfterRemovingCapturedStones(capturedStones)

        printDebugInfo(message: "After adding stone.")

        // Suicide is illegal.
        guard let newGroup = groups[newGroupID] else {
            fatalErrorForGroupsInvariance(groupID: newGroupID)
        }

        guard !newGroup.liberties.isEmpty else {
            throw IllegalMove.suicide
        }

        return capturedStones
    }

    private func checkLibertyGroupsInvariance() -> Bool {
        var groupIDsInGroupIndex = Set<Int>()
        let size = gameConfiguration.size
        for x in 0..<size {
            for y in 0..<size {
                guard let groupID = groupIndex[x][y] else {
                    continue
                }
                groupIDsInGroupIndex.insert(groupID)
            }
        }
        return Set(groups.keys) == groupIDsInGroupIndex
    }

    private func fatalErrorForGroupsInvariance(groupID: Int) -> Never {
        print("The group ID \(groupID) should exist.")
        print("Current groups are \(groups).")
        fatalError()
    }

    /// Returns the group index of the stone.
    private func groupIndex(for position: Position) -> Int? {
        return groupIndex[position.x][position.y]
    }

    /// Assigns a new unique group ID.
    mutating private func assignNewGroupID() -> Int {
        let newID = nextGroupIDToAssign
        precondition(groups[newID] == nil)

        nextGroupIDToAssign += 1
        return newID
    }

    /// Creates a new group for the single stone with liberties.
    mutating private func makeGroup(
        color: Color,
        stone: Position,
        liberties: Set<Position>
    ) -> LibertyGroup {
        let newID = assignNewGroupID()
        let newGroup = LibertyGroup(id: newID, color: color, stones: [stone], liberties: liberties)

        precondition(groups[newID] == nil)
        groups[newID] = newGroup
        groupIndex[stone.x][stone.y] = newID
        assert(checkLibertyGroupsInvariance())
        return newGroup
    }

    /// Returns a new group (id) by merging the groups identified by the IDs.
    mutating private func mergeGroups(_ groupID1: Int, _ groupID2: Int) -> Int {
        guard let group1 = groups.removeValue(forKey: groupID1) else {
            fatalErrorForGroupsInvariance(groupID: groupID1)
        }
        guard let group2 = groups.removeValue(forKey: groupID2) else {
            fatalErrorForGroupsInvariance(groupID: groupID2)
        }
        precondition(group1.color == group2.color)

        let newID = assignNewGroupID()

        let unionedStones = group1.stones.union(group2.stones)
        var newLiberties = group1.liberties.union(group2.liberties)
        newLiberties.subtract(group1.stones)
        newLiberties.subtract(group2.stones)

        let newGroup = LibertyGroup(
            id: newID,
            color: group1.color,
            stones: unionedStones,
            liberties: newLiberties
        )

        groups[newID] = newGroup

        // Updates groups IDs for future lookups.
        for stone in unionedStones {
            groupIndex[stone.x][stone.y] = newID
        }
        assert(checkLibertyGroupsInvariance())
        return newID
    }

    /// Captures the whole group and returns all stones in it.
    mutating private func captureGroup(_ groupID: Int) -> Set<Position> {
        let deadGroup = groups.removeValue(forKey: groupID)!
        for stone in deadGroup.stones {
            groupIndex[stone.x][stone.y] = nil
        }
        return deadGroup.stones
    }

    /// Updates all neighboring groups' liberties.
    mutating private func updateLibertiesAfterRemovingCapturedStones(_ capturedStones: Set<Position>) {
        let size = gameConfiguration.size
        for capturedStone in capturedStones {
            for neighbor in capturedStone.neighbors(boardSize: size) {
                if let neighborGroupdID = groupIndex(for: neighbor) {
                    guard groups.keys.contains(neighborGroupdID) else {
                        fatalErrorForGroupsInvariance(groupID:neighborGroupdID)
                    }
                    // This force unwrap is safe as we checked the key existence above. As
                    // the value in the groups is struct. We need the force unwrap to do
                    // mutation in place.
                    groups[neighborGroupdID]!.liberties.insert(capturedStone)
                }
            }
        }
        assert(checkLibertyGroupsInvariance())
    }

    /// Prints the debug info for liberty tracked so far.
    private func printDebugInfo(message: String) {
        guard gameConfiguration.isVerboseDebuggingEnabled else {
            return
        }

        print(message)

        /// Prints the group index for the board.
        let size = gameConfiguration.size
        for x in 0..<size {
            for y in 0..<size {
                switch groupIndex[x][y] {
                case .none:
                    print("  .", terminator: "")
                case .some(let id) where id < 10:
                    print("  \(id)", terminator: "")
                case .some(let id):
                    print(" \(id)", terminator: "")
                }
            }
            print("")
        }

        for (id, group) in groups {
            print(" id: \(id) -> liberty: \(group.liberties)")
        }
    }
}


