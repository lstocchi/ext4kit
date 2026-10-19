/*
 * Equivalent to `ext2fs_dirhash` in e2fsprogs for the `EXT2_HASH_HALF_MD4` algorithm:
 * https://github.com/tytso/e2fsprogs/blob/master/lib/ext2fs/dirhash.c
 */

pub const DX_HASH_HALF_MD4: u8 = 1;

const K1: u32 = 0;
const K2: u32 = 0x5a82_7999;
const K3: u32 = 0x6ed9_eba1;

#[inline(always)]
fn f(x: u32, y: u32, z: u32) -> u32 {
    (x & y) | (!x & z)
}

#[inline(always)]
fn g(x: u32, y: u32, z: u32) -> u32 {
    (x & y) | (x & z) | (y & z)
}

#[inline(always)]
fn h(x: u32, y: u32, z: u32) -> u32 {
    x ^ y ^ z
}

macro_rules! round {
    ($f:ident, $a:ident, $b:ident, $c:ident, $d:ident, $x:expr, $s:expr) => {
        $a = $a.wrapping_add($f($b, $c, $d)).wrapping_add($x);
        $a = $a.rotate_left($s);
    };
}

pub fn half_md4_transform(buf: &mut [u32; 4], in_buf: &[u32; 8]) {
    let mut a = buf[0];
    let mut b = buf[1];
    let mut c = buf[2];
    let mut d = buf[3];

    // Round 1
    round!(f, a, b, c, d, in_buf[0].wrapping_add(K1), 3);
    round!(f, d, a, b, c, in_buf[1].wrapping_add(K1), 7);
    round!(f, c, d, a, b, in_buf[2].wrapping_add(K1), 11);
    round!(f, b, c, d, a, in_buf[3].wrapping_add(K1), 19);
    round!(f, a, b, c, d, in_buf[4].wrapping_add(K1), 3);
    round!(f, d, a, b, c, in_buf[5].wrapping_add(K1), 7);
    round!(f, c, d, a, b, in_buf[6].wrapping_add(K1), 11);
    round!(f, b, c, d, a, in_buf[7].wrapping_add(K1), 19);

    // Round 2
    round!(g, a, b, c, d, in_buf[1].wrapping_add(K2), 3);
    round!(g, d, a, b, c, in_buf[3].wrapping_add(K2), 5);
    round!(g, c, d, a, b, in_buf[5].wrapping_add(K2), 9);
    round!(g, b, c, d, a, in_buf[7].wrapping_add(K2), 13);
    round!(g, a, b, c, d, in_buf[0].wrapping_add(K2), 3);
    round!(g, d, a, b, c, in_buf[2].wrapping_add(K2), 5);
    round!(g, c, d, a, b, in_buf[4].wrapping_add(K2), 9);
    round!(g, b, c, d, a, in_buf[6].wrapping_add(K2), 13);

    // Round 3
    round!(h, a, b, c, d, in_buf[3].wrapping_add(K3), 3);
    round!(h, d, a, b, c, in_buf[7].wrapping_add(K3), 9);
    round!(h, c, d, a, b, in_buf[2].wrapping_add(K3), 11);
    round!(h, b, c, d, a, in_buf[6].wrapping_add(K3), 15);
    round!(h, a, b, c, d, in_buf[1].wrapping_add(K3), 3);
    round!(h, d, a, b, c, in_buf[5].wrapping_add(K3), 9);
    round!(h, c, d, a, b, in_buf[0].wrapping_add(K3), 11);
    round!(h, b, c, d, a, in_buf[4].wrapping_add(K3), 15);

    buf[0] = buf[0].wrapping_add(a);
    buf[1] = buf[1].wrapping_add(b);
    buf[2] = buf[2].wrapping_add(c);
    buf[3] = buf[3].wrapping_add(d);
}

fn str2hashbuf(msg: &[u8], len: usize, buf: &mut [u32; 8], mut num: usize) {
    let pad_len = len as u32;
    let mut pad = pad_len | (pad_len << 8);
    pad |= pad << 16;

    let mut val = pad;
    let limit = len.min(num * 4);
    let mut buf_idx = 0;

    for i in 0..limit {
        // Sign-extend u8 -> i8 -> i32 -> u32 to match C's `(int)(signed char)c`
        let c = (msg[i] as i8 as i32) as u32;
        val = c.wrapping_add(val << 8);

        if i % 4 == 3 {
            buf[buf_idx] = val;
            buf_idx += 1;
            val = pad;
            num -= 1;
        }
    }

    if num > 0 {
        buf[buf_idx] = val;
        buf_idx += 1;
        num -= 1;
    }

    while num > 0 {
        buf[buf_idx] = pad;
        buf_idx += 1;
        num -= 1;
    }
}

pub fn ext4_dirhash_half_md4(name: &[u8], seed: Option<&[u32; 4]>) -> (u32, u32) {
    let mut buf = [0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32];

    // Check to see if the seed is provided and not all zeros
    if let Some(s) = seed {
        if s.iter().any(|&v| v != 0) {
            buf.copy_from_slice(s);
        }
    }

    let mut p_index = 0;
    let mut len = name.len();

    while len > 0 {
        let mut in_buf = [0u32; 8];
        str2hashbuf(&name[p_index..], len, &mut in_buf, 8);

        half_md4_transform(&mut buf, &in_buf);

        let chunk_len = len.min(32);
        len -= chunk_len;
        p_index += chunk_len;
    }

    let minor_hash = buf[2];
    let hash = buf[1];

    // ret_hash = hash & ~1
    (hash & !1, minor_hash)
}
