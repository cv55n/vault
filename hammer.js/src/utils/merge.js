import deprecate from './deprecate';
import extend from './extend';

/**
 * @private
 * 
 * significa que as propriedades que existem no dest não serão reescritas no src
 * 
 * @param {Object} dest
 * @param {Object} src
 * 
 * @returns {Object} dest
 */
const merge = deprecate((dest, src) => {
    return extend(dest, src, true);
}, 'merge', 'utilizar `assign`.');

export default merge;