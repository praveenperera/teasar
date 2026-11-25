//! The two supported settings sets and the one table of what each method accepts
//!
//! A call names its set; every other settings value it passes must equal that set's. The set is never inferred from
//! the values, so a configuration that matches a set's numbers but not its slab geometry cannot pass as that set

use std::fmt;

use crate::geom::{Block, SlabKind, SlabSpec};
use crate::{Error, Result};

/// The pre-skeleton settings: max-pool factor and the pooled size filter `min_count <= count < max_count`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrepSettings {
    pub downsample: u32,
    pub min_count: u32,
    pub max_count: u32,
}

/// The skeletonizer's settings as villa passes them
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkelSettings {
    pub anisotropy: [f64; 3],
    pub scale: f64,
    pub constant: f64,
    pub dust_threshold: u32,
    pub fix_borders: bool,
    pub fix_branching: bool,
}

/// The two settings sets v1 supports
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsSet {
    Published0211,
    VillaDefault,
}

/// A Python method that takes settings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    PrepareLabels,
    PrepareFromBlock,
    Skeletonize,
    Extract,
}

/// A settings argument other than `settings` itself, by its Python keyword
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Downsample,
    MinCount,
    MaxCount,
    Anisotropy,
    DustThreshold,
    Scale,
    Const,
    FixBranching,
    /// A free field: any bool, applied with villa's per-slab rule
    FixBorders,
    /// A free field, accepted with either set: `'best'`, the farthest-first order, or `'kimimaro'`,
    /// compatibility mode
    TargetOrder,
}

/// A settings value as the caller passed it, before it is checked
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Number(f64),
    /// Any sequence of numbers, so a wrong length is reported rather than a type error
    Numbers(Vec<f64>),
    Text(String),
    /// Anything else, by its repr; it never equals a set's value
    Other(String),
}

/// What one method accepts: the fields checked against the named set, and whether its slabs must fit the set's
/// geometry
#[derive(Debug)]
pub struct Accepted {
    pub method: Method,
    pub fields: &'static [Field],
    pub slab_thickness: bool,
}

const PREP_FIELDS: &[Field] = &[Field::Downsample, Field::MinCount, Field::MaxCount];
const SKEL_FIELDS: &[Field] = &[
    Field::Anisotropy,
    Field::DustThreshold,
    Field::Scale,
    Field::Const,
    Field::FixBranching,
    Field::FixBorders,
    Field::TargetOrder,
];

/// The accepted-settings table, the one contract the bindings and every backend apply; every method also
/// requires `settings` to name a set
pub const ACCEPTED: [Accepted; 4] = [
    Accepted { method: Method::PrepareLabels, fields: PREP_FIELDS, slab_thickness: false },
    Accepted { method: Method::PrepareFromBlock, fields: PREP_FIELDS, slab_thickness: true },
    Accepted { method: Method::Skeletonize, fields: SKEL_FIELDS, slab_thickness: false },
    Accepted {
        method: Method::Extract,
        fields: &[
            Field::Downsample,
            Field::MinCount,
            Field::MaxCount,
            Field::Anisotropy,
            Field::DustThreshold,
            Field::Scale,
            Field::Const,
            Field::FixBranching,
            Field::FixBorders,
            Field::TargetOrder,
        ],
        slab_thickness: true,
    },
];

impl Method {
    /// This method's row of [`ACCEPTED`]
    pub fn accepted(self) -> &'static Accepted {
        ACCEPTED.iter().find(|a| a.method == self).expect("every method has a row")
    }
}

impl Field {
    /// The Python keyword
    pub fn name(self) -> &'static str {
        match self {
            Field::Downsample => "downsample",
            Field::MinCount => "min_count",
            Field::MaxCount => "max_count",
            Field::Anisotropy => "anisotropy",
            Field::DustThreshold => "dust_threshold",
            Field::Scale => "scale",
            Field::Const => "const",
            Field::FixBranching => "fix_branching",
            Field::FixBorders => "fix_borders",
            Field::TargetOrder => "target_order",
        }
    }
}

impl Value {
    /// Numeric equality, so `2` equals `2.0`; a bool equals only a bool
    fn equals(&self, want: &Value) -> bool {
        match (self, want) {
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Number(a), Value::Number(b)) => a == b,
            (Value::Numbers(a), Value::Numbers(b)) => a == b,
            _ => false,
        }
    }
}

/// Values print as Python would show them, since the messages reach Python callers
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Bool(true) => f.write_str("True"),
            Value::Bool(false) => f.write_str("False"),
            Value::Number(x) => write!(f, "{x}"),
            Value::Numbers(xs) => {
                let parts: Vec<String> = xs.iter().map(f64::to_string).collect();
                write!(f, "({})", parts.join(", "))
            }
            Value::Text(t) => write!(f, "'{t}'"),
            Value::Other(repr) => f.write_str(repr),
        }
    }
}

/// The free fields' rules: fix_borders takes any bool, target_order `'best'` or `'kimimaro'`
fn check_free(field: Field, got: &Value) -> Result<()> {
    let why = match (field, got) {
        (Field::FixBorders, Value::Bool(_)) => return Ok(()),
        (Field::FixBorders, _) => format!("fix_borders must be True or False, got {got}"),
        (Field::TargetOrder, Value::Text(t)) if t == "best" || t == "kimimaro" => return Ok(()),
        (Field::TargetOrder, _) => format!("target_order must be 'best' or 'kimimaro', got {got}"),
        _ => unreachable!("{field:?} is not a free field"),
    };
    Err(Error::UnsupportedSettings(why))
}

impl SettingsSet {
    pub const ALL: [SettingsSet; 2] = [SettingsSet::Published0211, SettingsSet::VillaDefault];

    /// The set named `name`, as `settings=` passes it
    pub fn parse(name: &str) -> Result<Self> {
        Self::ALL.into_iter().find(|s| s.name() == name).ok_or_else(|| {
            Error::UnsupportedSettings(format!(
                "settings={name:?} is not a supported settings set; use 'published_0211' or 'villa_default'"
            ))
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            SettingsSet::Published0211 => "published_0211",
            SettingsSet::VillaDefault => "villa_default",
        }
    }

    pub fn downsample(self) -> u32 {
        match self {
            SettingsSet::Published0211 => 2,
            SettingsSet::VillaDefault => 4,
        }
    }

    /// villa's formulas: `min_count = max(1, 400 // ds**3)`, `max_count = max(1, 640000 // ds**3)`
    pub fn prep(self) -> PrepSettings {
        let cube = self.downsample().pow(3);
        PrepSettings {
            downsample: self.downsample(),
            min_count: (400 / cube).max(1),
            max_count: (640_000 / cube).max(1),
        }
    }

    /// The skeleton settings villa passes with this set: scale 1, const 2, fix_branching on, and the caller's
    /// fix_borders
    pub fn skel_settings(self, fix_borders: bool) -> SkelSettings {
        SkelSettings {
            anisotropy: self.anisotropy(),
            scale: 1.0,
            constant: 2.0,
            dust_threshold: self.dust_threshold(),
            fix_borders,
            fix_branching: true,
        }
    }

    /// `dust_threshold = min_count - 1`
    pub fn dust_threshold(self) -> u32 {
        self.prep().min_count - 1
    }

    /// `anisotropy = (ds, ds, ds)`
    pub fn anisotropy(self) -> [f64; 3] {
        [f64::from(self.downsample()); 3]
    }

    /// villa's slab depth or thickness for `kind`: `h` ribbons 2 or 4 deep, vertical slabs 4 thick in both sets
    pub fn slab_thickness(self, kind: SlabKind) -> usize {
        match (self, kind) {
            (SettingsSet::Published0211, SlabKind::H) => 2,
            (SettingsSet::VillaDefault, SlabKind::H) => 4,
            (_, SlabKind::Vy | SlabKind::Vx) => 4,
        }
    }

    /// The set's value of `field`, or `None` for the free fields fix_borders and target_order
    fn value(self, field: Field) -> Option<Value> {
        let skel = self.skel_settings(true);
        Some(match field {
            Field::Downsample => Value::Number(f64::from(self.downsample())),
            Field::MinCount => Value::Number(f64::from(self.prep().min_count)),
            Field::MaxCount => Value::Number(f64::from(self.prep().max_count)),
            Field::Anisotropy => Value::Numbers(skel.anisotropy.to_vec()),
            Field::DustThreshold => Value::Number(f64::from(skel.dust_threshold)),
            Field::Scale => Value::Number(skel.scale),
            Field::Const => Value::Number(skel.constant),
            Field::FixBranching => Value::Bool(skel.fix_branching),
            Field::FixBorders | Field::TargetOrder => return None,
        })
    }

    /// Checks one call's settings against its row of [`ACCEPTED`] and returns the named set
    ///
    /// `given` holds every field of the row, each once, as the binding received it. A binding that passes a
    /// different set of fields is a bug in the binding, so that panics instead of returning an error
    pub fn accept(method: Method, name: &str, given: &[(Field, Value)]) -> Result<Self> {
        let set = Self::parse(name)?;
        let fields = method.accepted().fields;
        assert!(
            given.len() == fields.len() && fields.iter().all(|f| given.iter().any(|(g, _)| g == f)),
            "the {method:?} binding passed {:?}, but the table lists {fields:?}",
            given.iter().map(|(f, _)| *f).collect::<Vec<_>>()
        );

        for (field, got) in given {
            set.check_field(*field, got)?;
        }

        Ok(set)
    }

    fn check_field(self, field: Field, got: &Value) -> Result<()> {
        let Some(want) = self.value(field) else {
            return check_free(field, got);
        };
        if got.equals(&want) {
            return Ok(());
        }
        Err(Error::UnsupportedSettings(format!(
            "settings='{}' needs {name}={want}, got {name}={got}",
            self.name(),
            name = field.name()
        )))
    }

    /// Checks that `slab` fits the set's slab geometry: its thickness is the set's, or less where the slab ends at
    /// the block's edge on its axis
    pub fn check_slab(self, block: &Block<'_>, slab: &SlabSpec) -> Result<()> {
        let want = self.slab_thickness(slab.kind);
        let got = (slab.hi - slab.lo) as usize;
        let at_edge = slab.hi as usize == block.axis_len(slab.kind);
        if got == want || (at_edge && got < want) {
            return Ok(());
        }
        Err(Error::UnsupportedSettings(format!(
            "settings='{}' cuts {} slabs {want} thick, got {slab} ({got} thick{})",
            self.name(),
            slab.kind.name(),
            if at_edge { ", ending at the block's edge" } else { "" }
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sets_follow_villas_formulas() {
        let p = SettingsSet::Published0211.prep();
        assert_eq!((p.downsample, p.min_count, p.max_count), (2, 50, 80_000));
        assert_eq!(SettingsSet::Published0211.dust_threshold(), 49);
        let v = SettingsSet::VillaDefault.prep();
        assert_eq!((v.downsample, v.min_count, v.max_count), (4, 6, 10_000));
        assert_eq!(SettingsSet::VillaDefault.dust_threshold(), 5);
    }
}
